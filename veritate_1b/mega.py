# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - MEGA model: vanilla Veritate with the FFN in every block replaced by a
#   top-1 mixture-of-experts FFN. Same attention path, same embeddings, same
#   RMSNorm pre-norm, same tied LM head. Only the FFN differs.
# - Top-1 routing only (matches the v11 engine spec in
#   documentation/kernels/moe.md). Router probabilities are scaled into the
#   expert output so the router gets gradient via Switch-Transformer style.
# - Auxiliary load-balance loss is computed inside forward and returned alongside
#   the LM loss. The trainer adds it with a coefficient.
# - hook_spec() returns a freshly-instantiated vanilla Veritate sized to match
#   this MEGA's hidden/layers/heads/seq, with each block's FFN populated from
#   the corresponding block's expert-0. The dashboard's dump suite walks that
#   adapter so probe / lens / classroom / etc. continue to render. The MEGA
#   checkpoint itself contains the full multi-expert state_dict.
# plugins/veritate_mega/mega.py
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from veritate_core import qat as _qat
from veritate_core.model import (
    VOCAB_BYTE_LEVEL,
    RMSNorm,
    QuantLinear,
    CausalSelfAttention,
    Veritate,
)


class MoEFFN(nn.Module):
    """Top-1 MoE FFN. router . x -> n_experts logits, argmax picks the expert,
    selected expert's FFN runs on each token, output is scaled by the router's
    softmax probability for the chosen expert so the router receives gradient."""

    def __init__(self, hidden, ffn, n_experts):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        self.hidden    = hidden
        self.ffn       = ffn
        self.n_experts = n_experts
        self.router    = QuantLinear(hidden, n_experts, bias=False)
        self.experts_up   = nn.ModuleList([QuantLinear(hidden, ffn,    bias=False) for _ in range(n_experts)])
        self.experts_down = nn.ModuleList([QuantLinear(ffn,    hidden, bias=False) for _ in range(n_experts)])
        self.qat = False

    def forward(self, x):
        # Returns (out, aux_loss). aux_loss is fp32 during training, None at eval.
        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)

        logits = self.router(x_flat)
        scores = F.softmax(logits.float(), dim=-1)
        top_idx = scores.argmax(dim=-1)
        top_p   = scores.gather(1, top_idx.unsqueeze(1)).squeeze(1).to(x.dtype)

        # Sort tokens by expert assignment so each expert sees a contiguous
        # slab. One device->host sync per layer (the per-expert counts) instead
        # of n_experts syncs from .nonzero() in a Python loop.
        sort_idx = torch.argsort(top_idx, stable=True)
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(sort_idx.numel(), device=x.device)

        x_sorted     = x_flat.index_select(0, sort_idx)
        top_p_sorted = top_p.index_select(0, sort_idx)
        counts = torch.bincount(top_idx, minlength=self.n_experts).tolist()

        out_sorted = torch.empty_like(x_flat)
        offset = 0
        for e in range(self.n_experts):
            cnt = counts[e]
            if cnt == 0:
                continue
            slab = x_sorted.narrow(0, offset, cnt)
            slab = self.experts_down[e](F.gelu(self.experts_up[e](slab)))
            out_sorted.narrow(0, offset, cnt).copy_(slab * top_p_sorted.narrow(0, offset, cnt).unsqueeze(-1))
            offset += cnt

        out = out_sorted.index_select(0, inv_sort)

        aux = None
        if self.training:
            f_e = torch.bincount(top_idx, minlength=self.n_experts).to(scores.dtype) / float(B * T)
            P_e = scores.mean(dim=0)
            aux = self.n_experts * (f_e * P_e).sum()

        return out.reshape(B, T, H), aux


class MegaBlock(nn.Module):
    def __init__(self, hidden, ffn, heads, n_experts):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads)
        self.n2   = RMSNorm(hidden)
        self.ff   = MoEFFN(hidden, ffn, n_experts)
        self.qat  = False

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.qat: x = _qat.fake_quant_act(x)
        ff_out, aux = self.ff(self.n2(x))
        x = x + ff_out
        if self.qat: x = _qat.fake_quant_act(x)
        return x, aux


class Mega(nn.Module):
    """MoE Veritate. State dict contains all expert weights; hook_spec() returns
    a vanilla Veritate populated with each block's expert-0 for the dump suite."""

    def __init__(self, vocab, hidden, layers, ffn, heads, seq, n_experts, router_topk=1):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        if router_topk != 1:
            raise ValueError(f"only router_topk=1 is supported (got {router_topk}); top-K reserved")

        self.vocab       = vocab
        self.hidden      = hidden
        self.layers      = layers
        self.ffn         = ffn
        self.heads       = heads
        self.seq         = seq
        self.n_experts   = n_experts
        self.router_topk = router_topk
        self.qat         = False

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq,   hidden)
        self.blocks  = nn.ModuleList([MegaBlock(hidden, ffn, heads, n_experts) for _ in range(layers)])
        self.n_out   = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # Avoid re-creating torch.arange(T) on every forward.
        self.register_buffer("pos_ids", torch.arange(seq), persistent=False)

        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def set_qat(self, value):
        return _qat.set_qat(self, value)

    def embed(self, tokens):
        B, T = tokens.shape
        if T > self.seq:
            raise ValueError(f"input length {T} exceeds seq {self.seq}")
        pos = self.pos_ids[:T]
        e = self.tok_emb(tokens) + self.pos_emb(pos)
        if self.qat: e = _qat.fake_quant_act(e)
        return e

    def forward(self, tokens, targets=None):
        """Returns (logits, lm_loss, aux_loss). aux_loss is None at eval."""
        x = self.embed(tokens)
        aux_terms = []
        for blk in self.blocks:
            x, aux = blk(x)
            if aux is not None:
                aux_terms.append(aux)
        x = self.n_out(x)
        logits = self.lm_head(x)
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        aux_loss = torch.stack(aux_terms).mean() if aux_terms else None
        return logits, lm_loss, aux_loss

    def hook_spec(self):
        """Return a vanilla Veritate snapshot of this MEGA, populated with each
        block's expert-0 weights and the shared attention / embedding / norm /
        head. Used by the dashboard dump suite, not for training or inference.
        Snapshot weights are deliberately copied with no_grad on CPU; the
        dump suite moves them to the right device itself."""
        with torch.no_grad():
            # Skip the constructor's random initialization; every parameter is
            # overwritten in this method anyway. Saves ~1-2s per checkpoint at
            # the 1B size.
            with torch.device("meta"):
                view = Veritate(
                    vocab=self.vocab, hidden=self.hidden, layers=self.layers,
                    ffn=self.ffn, heads=self.heads, seq=self.seq,
                )
            view = view.to_empty(device="cpu")
            view.tok_emb.weight.copy_(self.tok_emb.weight.detach().cpu())
            view.pos_emb.weight.copy_(self.pos_emb.weight.detach().cpu())
            view.n_out.weight.copy_(self.n_out.weight.detach().cpu())
            for vblk, mblk in zip(view.blocks, self.blocks):
                vblk.n1.weight.copy_(mblk.n1.weight.detach().cpu())
                vblk.n2.weight.copy_(mblk.n2.weight.detach().cpu())
                vblk.attn.qkv.weight.copy_(mblk.attn.qkv.weight.detach().cpu())
                vblk.attn.proj.weight.copy_(mblk.attn.proj.weight.detach().cpu())
                vblk.ff.up.weight.copy_(mblk.ff.experts_up[0].weight.detach().cpu())
                vblk.ff.down.weight.copy_(mblk.ff.experts_down[0].weight.detach().cpu())
        return view
