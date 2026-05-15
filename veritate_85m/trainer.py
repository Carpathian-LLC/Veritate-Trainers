# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_85m: byte-level decoder at h=768 L=12 ffn=3072 heads=12 (vocab=256).
#   ReLU FFN with L1 sparsity loss on post-activation, 2-byte multi-token-
#   prediction (MTP) head, byte0_projector contract method for cross-model
#   decode (per preflight rule 11a).
# - Optimization stack: bf16 autocast on CUDA (fp32 on CPU), foreach AdamW,
#   WSD schedule with sqrt decay last 10%, NaN-skip guard, bf16-native RMSNorm.
# - Uses save.save() so checkpoints carry the full dashboard hook suite (probe,
#   classroom, grades, math, grammar, reasoning, concepts, surprise, quant_kl,
#   writing_health, generation, reading_comprehension) and save.append_train_row()
#   so train.csv lands in the canonical models/<name>/ location.
# - Corpus: plugins/corpus/fineweb_edu_train.bin (built by build_corpus.py).
# plugins/veritate_85m/plugin.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from veritate_core.plugin import save, paths  # noqa: E402
from veritate_core import qat as _qat  # noqa: E402
from veritate_core.model import QuantLinear, RMSNorm  # noqa: E402

with open(os.path.join(HERE, "manifest.json"), "r", encoding="utf-8") as _f:
    MANIFEST = json.load(_f)


# ------------------------------------------------------------------------------------
# Constants

VOCAB_BYTE_LEVEL = 256
LR_SCHEDULES = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS = ("sqrt", "linear", "cosine")
PRECISIONS = ("fp32", "bf16")
BASE_CKPT_PREFIX = "step_"
BASE_CKPT_SUFFIX = ".pt"


# ------------------------------------------------------------------------------------
# Model

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.h = heads
        self.d = hidden // heads
        self.qkv = QuantLinear(hidden, 3 * hidden, bias=False)
        self.proj = QuantLinear(hidden, hidden, bias=False)
        self.qat = False
        self.engine_faithful = False

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qat and self.engine_faithful:
            q = _qat.fake_quant_act(q)
            k = _qat.fake_quant_act(k)
            v = _qat.fake_quant_act(v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.qat and self.engine_faithful:
            out = _qat.fake_quant_act(out)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class ReluFFN(nn.Module):
    """FFN with ReLU activation. Stores post-ReLU L1 mean as an attribute so
    the L1 sparsity penalty stays inside the autograd graph (model.forward
    sums across blocks) without changing the Block.forward signature that
    cross-model decoders rely on (preflight rule 11a)."""

    def __init__(self, hidden, ffn):
        super().__init__()
        self.up = QuantLinear(hidden, ffn, bias=False)
        self.down = QuantLinear(ffn, hidden, bias=False)
        self._last_l1 = None

    def forward(self, x):
        post = F.relu(self.up(x))
        self._last_l1 = post.abs().mean()
        return self.down(post)


class Block(nn.Module):
    def __init__(self, hidden, ffn, heads):
        super().__init__()
        self.n1 = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads)
        self.n2 = RMSNorm(hidden)
        self.ff = ReluFFN(hidden, ffn)
        self.qat = False

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.qat:
            x = _qat.fake_quant_act(x)
        x = x + self.ff(self.n2(x))
        if self.qat:
            x = _qat.fake_quant_act(x)
        return x


class MTPHead(nn.Module):
    """N independent prediction heads off the final hidden state. Head 0 is
    identity-initialized so byte-0 inherits the trunk's existing semantics.
    Heads 1..N-1 start near-zero and learn future-byte prediction."""

    def __init__(self, hidden, n_predict, lm_head):
        super().__init__()
        self.n_predict = n_predict
        self.transforms = nn.ModuleList([
            QuantLinear(hidden, hidden, bias=False) for _ in range(n_predict)
        ])
        with torch.no_grad():
            self.transforms[0].weight.copy_(torch.eye(hidden))
            for i in range(1, n_predict):
                nn.init.normal_(self.transforms[i].weight, mean=0.0, std=0.01)
        self.norms = nn.ModuleList([RMSNorm(hidden) for _ in range(n_predict)])
        self.lm_head = lm_head

    def forward(self, h):
        outs = []
        for i in range(self.n_predict):
            outs.append(self.lm_head(self.norms[i](self.transforms[i](h))))
        return torch.stack(outs, dim=2)  # [B, T, N, vocab]


class Veritate85M(nn.Module):
    """Byte-level h=768 L=12 trunk. ReLU+L1 FFN + 2-byte MTP head."""

    def __init__(self, vocab, hidden, layers, ffn, heads, seq, n_predict=2):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.vocab = vocab
        self.hidden = hidden
        self.layers = layers
        self.ffn = ffn
        self.heads = heads
        self.seq = seq
        self.n_predict = n_predict
        self._mtp_aux_weight = 0.5

        self.qat = False
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq, hidden)
        self.blocks = nn.ModuleList([Block(hidden, ffn, heads) for _ in range(layers)])
        self.n_out = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        # Untied to avoid the MTP autograd-accumulation pitfall the 800M trunk hit.

        self.mtp = MTPHead(hidden, n_predict, self.lm_head)

        # Initialize: all linears + tok_emb get N(0, 0.02), skip MTP transforms
        # (already initialized in MTPHead.__init__).
        mtp_param_ids = {id(t.weight) for t in self.mtp.transforms}
        for p in self.parameters():
            if p.dim() >= 2 and id(p) not in mtp_param_ids:
                nn.init.normal_(p, mean=0.0, std=0.02)

    # ------------ cross-model contract (preflight rule 11a) ------------

    def project_byte0(self, residual):
        h = self.n_out(residual)
        return self.lm_head(self.mtp.norms[0](self.mtp.transforms[0](h)))

    def ensure_context(self, T):
        if T > self.seq:
            raise ValueError(f"input length {T} exceeds seq {self.seq}")

    def run_blocks(self, x, start_pos=0, exit_after=None):
        n = self.layers if exit_after is None else min(int(exit_after), self.layers)
        for L in range(n):
            x = self.run_block(x, L, start_pos=start_pos)
        return x

    def run_block(self, x, L, start_pos=0):
        return self.blocks[L](x)

    def supports_mtp_decode(self):
        return int(self.n_predict) > 1

    def embed(self, tokens, start_pos=0):
        B, T = tokens.shape
        if start_pos + T > self.seq:
            raise ValueError(f"start_pos+T ({start_pos + T}) exceeds seq {self.seq}")
        pos = torch.arange(start_pos, start_pos + T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        if self.qat:
            x = _qat.fake_quant_act(x)
        return x

    def kv_cache_patch_attn(self, attn_mod, cache, get_start_pos):
        def fwd(x):
            B, T, C = x.shape
            h, d = attn_mod.h, attn_mod.d
            qkv = attn_mod.qkv(x).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            cache.append(k, v)
            K, V = cache.view()
            if T == 1:
                out = F.scaled_dot_product_attention(q, K, V, is_causal=False)
            else:
                S = K.size(2)
                abs_start = cache.length - T
                i_idx = torch.arange(T, device=x.device).view(T, 1) + abs_start
                j_idx = torch.arange(S, device=x.device).view(1, S)
                mask = (j_idx <= i_idx)
                attn_mask = torch.zeros(T, S, device=x.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
                out = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask, is_causal=False)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return attn_mod.proj(out)
        return fwd

    # ------------ forward ------------

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        self.ensure_context(T)
        x = self.embed(tokens)
        x = self.run_blocks(x)
        l1_mean = sum(blk.ff._last_l1 for blk in self.blocks) if targets is not None else None
        h = self.n_out(x)

        all_logits = self.mtp(h)  # [B, T, N, vocab]
        logits_b0 = all_logits[:, :, 0, :]

        total_loss = None
        if targets is not None:
            T_keep = max(1, T - (self.n_predict - 1))
            head_targets = torch.empty(B, T_keep, self.n_predict, dtype=targets.dtype, device=targets.device)
            head_targets[:, :, 0] = targets[:, :T_keep]
            for i in range(1, self.n_predict):
                head_targets[:, :, i] = targets[:, i:T_keep + i]
            aux_w = float(self._mtp_aux_weight)
            weights = [1.0] + [aux_w] * (self.n_predict - 1)
            losses = []
            for i in range(self.n_predict):
                logits_i = all_logits[:, :T_keep, i, :].reshape(-1, self.vocab)
                tgt_i = head_targets[:, :, i].reshape(-1)
                losses.append(F.cross_entropy(logits_i, tgt_i))
            total_loss = sum(weights[i] * losses[i] for i in range(self.n_predict))

        return logits_b0, total_loss, l1_mean


# ------------------------------------------------------------------------------------
# Helpers

def lr_at(step, total, warmup, base_lr, min_lr, schedule="wsd",
          wsd_decay_frac=0.1, wsd_decay_kind="sqrt"):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return base_lr + (min_lr - base_lr) * p
    if schedule == "wsd":
        decay_frac = max(1e-6, min(1.0, float(wsd_decay_frac)))
        stable_p = 1.0 - decay_frac
        if p <= stable_p:
            return base_lr
        q = (p - stable_p) / decay_frac
        if wsd_decay_kind == "linear":
            shape = 1.0 - q
        elif wsd_decay_kind == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * q))
        else:
            shape = 1.0 - math.sqrt(q)
        return min_lr + (base_lr - min_lr) * shape
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def pick_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_data_loader(bin_path, total_chunk_len, batch_size, seed, pin=False, prefetch=0,
                     in_ram=False):
    """Producer-consumer dataloader. `pin=True` returns pinned-host tensors so
    `.to(device, non_blocking=True)` actually overlaps. `prefetch>0` runs a
    background thread that fills the next N batches while the GPU is busy with
    the current step. `in_ram=True` loads the whole corpus into a contiguous
    numpy array — random access then never hits disk. Determinism preserved
    (single producer, single consumer)."""
    if in_ram:
        # 5 GB fineweb fits easily in 32 GB system RAM; eliminates disk I/O
        # variance that otherwise caps tok/s at the page-cache hit rate.
        arr = np.fromfile(bin_path, dtype=np.uint8)
    else:
        arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    N = len(arr)
    rng = np.random.RandomState(seed)
    if N < total_chunk_len + 2:
        raise ValueError(f"corpus too small for chunk length: {N} < {total_chunk_len + 2}")

    def _draw_one():
        starts = rng.randint(0, N - total_chunk_len - 1, size=batch_size, dtype=np.int64)
        toks = np.empty((batch_size, total_chunk_len), dtype=np.int64)
        tgts = np.empty((batch_size, total_chunk_len), dtype=np.int64)
        for b, s in enumerate(starts):
            toks[b] = arr[s:s + total_chunk_len]
            tgts[b] = arr[s + 1:s + 1 + total_chunk_len]
        t_toks = torch.from_numpy(toks)
        t_tgts = torch.from_numpy(tgts)
        if pin:
            t_toks = t_toks.pin_memory()
            t_tgts = t_tgts.pin_memory()
        return t_toks, t_tgts

    if prefetch <= 0:
        return _draw_one, N

    q = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def _producer():
        while not stop.is_set():
            try:
                q.put(_draw_one(), timeout=1.0)
            except queue.Full:
                continue

    t = threading.Thread(target=_producer, name="data-prefetch", daemon=True)
    t.start()

    def draw():
        return q.get()

    return draw, N


def apply_resume_overrides(args, argv):
    """Restore the original training_args from the resumed model's config.json
    for any flag the caller did NOT pass on the CLI. Matches the contract used
    by the 800m / 1b / 200m / 80m plugins so a continue-training run picks up
    the right corpus / shape / schedule by default, while still letting the
    user explicitly override (e.g. swap corpus, extend total_steps)."""
    cfg_path = paths.config_path(args.resume)
    if not os.path.isfile(cfg_path):
        return  # no config — nothing to restore; fall back to CLI / manifest defaults
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return
    ta = cfg.get("training_args") or {}
    for k, v in ta.items():
        if not hasattr(args, k):
            continue
        flag = "--" + k
        # If the user passed this flag on the CLI (with value or as bare arg),
        # respect the explicit override and do not replay the config value.
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            continue
        cur = getattr(args, k)
        if isinstance(cur, bool) and not isinstance(v, bool):
            continue
        try:
            setattr(args, k, type(cur)(v) if cur is not None else v)
        except (TypeError, ValueError):
            setattr(args, k, v)


def latest_checkpoint_step(model_dir):
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("no checkpoints dir for: " + model_dir)
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(BASE_CKPT_PREFIX) and fn.endswith(BASE_CKPT_SUFFIX):
            try:
                n = int(fn[len(BASE_CKPT_PREFIX):-len(BASE_CKPT_SUFFIX)])
            except ValueError:
                continue
            # Skip obvious junk: zero-byte / truncated stubs from a crash
            # mid-save before atomic-write landed. A real 85M bf16 checkpoint
            # is ~170MB; <100KB is always a partial torch.save preamble.
            try:
                if os.path.getsize(os.path.join(ckpt_dir, fn)) < 100_000:
                    continue
            except OSError:
                continue
            steps.append(n)
    if not steps:
        raise FileNotFoundError("no step_*.pt under: " + ckpt_dir)
    return max(steps)


# ------------------------------------------------------------------------------------
# Args

def parse_args():
    ap = argparse.ArgumentParser(description=MANIFEST.get("description", ""))
    # Path overrides. Default to "" so the resolver in main() picks paths from
    # --corpus <stem> via the shared corpus reader. Pass explicit paths only
    # when running outside the dashboard or pointing at a custom .bin.
    ap.add_argument("--corpus_bin", type=str, default="",
                    help="Override training .bin path. If empty (default), resolved "
                         "from --corpus stem via the shared corpus reader.")
    ap.add_argument("--val_bin", type=str, default="",
                    help="Override val .bin path. If empty (default), resolved from "
                         "--corpus stem via the shared corpus reader.")
    ap.add_argument("--output_dir", type=str,
                    default=os.path.join(REPO_ROOT, "models", "fineweb_edu_85m_bf16_v1_sparse"),
                    help="Output model directory. Ignored when --name is set.")
    ap.add_argument("--name", type=str, default="",
                    help="User-friendly model slug. When set, output_dir becomes "
                         "<repo>/models/<slug>_<size>, overriding --output_dir.")
    ap.add_argument("--device", type=str, default="auto",
                    choices=("auto", "cpu", "cuda", "mps"),
                    help="Force a device; 'auto' picks the best available.")
    ap.add_argument("--description", type=str, default="")
    ap.add_argument("--resume", type=str, default="",
                    help="Model name to resume (e.g. 'fineweb_edu_85m_bf16_v1_sparse'). "
                         "When set, output_dir becomes <repo>/models/<resume>, overriding "
                         "--output_dir and --name. Matches the other plugin trainers' contract.")
    for k, v in MANIFEST.get("defaults", {}).items():
        if isinstance(v, bool):
            ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(v))
        elif isinstance(v, int):
            ap.add_argument("--" + k, type=int, default=v)
        elif isinstance(v, float):
            ap.add_argument("--" + k, type=float, default=v)
        else:
            ap.add_argument("--" + k, type=str, default=str(v))
    # parse_known_args so the dashboard can send standard-schema flags this
    # plugin doesn't implement (e.g. --qat_enabled on a non-QAT trainer) and
    # they get silently dropped instead of crashing argparse. The dashboard
    # schema is the source of truth for which fields render; the manifest
    # only supplies pre-filled defaults.
    args, _ = ap.parse_known_args()
    return args


# ------------------------------------------------------------------------------------
# Main

def main():
    args = parse_args()

    # On --resume, replay the resumed model's training_args from config.json
    # for any flag the user did not pass on this CLI. Without this, a blank
    # corpus picker (the dashboard's "keep the original" affordance) would
    # silently fall through to the manifest default and swap corpora.
    if args.resume and args.resume.strip():
        apply_resume_overrides(args, sys.argv)

    if args.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr_schedule: {args.lr_schedule}")
    if args.lr_schedule == "wsd" and args.wsd_decay_kind not in WSD_DECAY_KINDS:
        raise ValueError(f"unknown wsd_decay_kind: {args.wsd_decay_kind}")
    if args.precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {args.precision}")

    # Resolve corpus stem → train/val .bin paths via the shared reader, unless
    # the caller passed explicit overrides. Centralizes "where do corpora live"
    # so the 85m honors the dashboard's corpus picker just like 800m/200m/etc.
    if not args.corpus_bin or not args.val_bin:
        stem = (getattr(args, "corpus", "") or "").strip()
        if not stem:
            raise ValueError("no --corpus stem and no --corpus_bin override")
        resolved_train, resolved_val = save.resolve_corpus(stem)
        if not args.corpus_bin:
            args.corpus_bin = resolved_train
        if not args.val_bin and resolved_val:
            args.val_bin = resolved_val

    if not os.path.isfile(args.corpus_bin):
        raise FileNotFoundError(
            f"corpus_bin not found: {args.corpus_bin}\n"
            f"Run: python plugins/veritate_85m/build_corpus.py"
        )

    device_type = pick_device(args.device)
    device = torch.device(device_type)
    amp_dtype = torch.bfloat16 if (args.precision == "bf16" and device_type == "cuda") else None
    # Enable TF32 for the fp32 paths that survive bf16 autocast (RMSNorm's
    # x.float() reduction is the main one). Free 5-15% on Ampere+ tensor cores.
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Path precedence: --resume wins (it's the existing model dir), then --name
    # (user composes a fresh one), then --output_dir (the manual override).
    if args.resume and args.resume.strip():
        args.output_dir = os.path.join(REPO_ROOT, "models", args.resume.strip())
    elif args.name and args.name.strip():
        composed = save.compose_name(args.name, args.size)
        args.output_dir = os.path.join(REPO_ROOT, "models", composed)
    os.makedirs(args.output_dir, exist_ok=True)
    # name is the basename of output_dir; save.save / save.append_train_row resolve
    # paths via veritate_mri/readers/paths.py, which roots at <repo>/models/<name>.
    name = os.path.basename(os.path.normpath(args.output_dir))

    qat_on = bool(getattr(args, "qat_enabled", False))
    quant_mode = getattr(args, "quant_mode", _qat.QUANT_MODE_INT8) or _qat.QUANT_MODE_INT8
    if qat_on and quant_mode not in _qat.QUANT_MODES:
        raise ValueError(f"unknown quant_mode: {quant_mode!r}; expected one of {_qat.QUANT_MODES}")
    cfg = {
        "vocab": args.vocab,
        "hidden": args.hidden,
        "layers": args.layers,
        "ffn": args.ffn,
        "heads": args.heads,
        "seq": args.seq,
        "n_predict": args.n_predict,
        "activation": "relu",
        "byte0_projector": "mtp.norms[0] o mtp.transforms[0] o lm_head",
        "training": ("qat" if qat_on else ""),
        "quant_mode": (quant_mode if qat_on else ""),
    }

    print(f"[veritate_85m] device={device_type} amp_dtype={amp_dtype}", flush=True)
    print(f"[veritate_85m] output={args.output_dir}", flush=True)
    print(f"[veritate_85m] corpus={args.corpus_bin}", flush=True)

    # Model
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Veritate85M(
        vocab=args.vocab, hidden=args.hidden, layers=args.layers,
        ffn=args.ffn, heads=args.heads, seq=args.seq, n_predict=args.n_predict,
    )
    model._mtp_aux_weight = args.mtp_aux_weight
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[veritate_85m] params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    # QAT: when args.qat_enabled is True, flip every submodule's .qat flag so
    # the next forward applies fake-quant. quant_mode picks weight scheme
    # (int8 / int4 / ternary). engine_faithful additionally fake-quants
    # q/k/v/out inside attention for engine-bitwise-match training.
    if qat_on:
        _qat.set_qat(model, True)
        _qat.set_quant_mode(model, quant_mode)
        if getattr(args, "engine_faithful", False):
            _qat.set_engine_faithful(model, True)
        print(f"[veritate_85m] QAT enabled: quant_mode={quant_mode} "
              f"engine_faithful={bool(getattr(args, 'engine_faithful', False))}",
              flush=True)

    # Optimizer. Fused AdamW is ~5-10% faster than foreach on CUDA; falls back
    # to foreach on CPU/MPS where fused is unsupported.
    _use_fused = (device_type == "cuda")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        betas=(args.beta1, args.beta2),
        eps=1e-6,
        weight_decay=args.weight_decay,
        fused=_use_fused,
        foreach=(not _use_fused),
    )

    # Resume
    start_step = 0
    if args.resume:
        last = latest_checkpoint_step(args.output_dir)
        ckpt = torch.load(
            os.path.join(args.output_dir, "checkpoints", f"step_{last}.pt"),
            map_location=device, weights_only=False,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", last)
        # Drop CSV rows past the checkpoint. Steps logged between the last .pt
        # and the crash never reached disk and will be retrained; leaving them
        # would put duplicate step numbers in the file.
        dropped = save.truncate_train_csv_at(name, start_step)
        print(f"[veritate_85m] resumed from step {start_step}"
              + (f" (dropped {dropped} stale CSV row(s))" if dropped else ""),
              flush=True)

    # config.json is written by save.save() via _ensure_config the first time
    # save.save() runs for this model dir; no need to write it here.

    # Data. CUDA path uses pinned memory + prefetch + in-RAM corpus so the
    # next batch is being prepared while the GPU runs the current step and
    # every random-offset read is RAM-fast.
    _pin = (device_type == "cuda")
    _prefetch = 4 if _pin else 0
    _in_ram = (device_type == "cuda")
    print(f"[veritate_85m] dataloader: pin={_pin} prefetch={_prefetch} in_ram={_in_ram}", flush=True)
    train_draw, n_train = make_data_loader(args.corpus_bin, args.seq, args.batch_size, args.seed,
                                           pin=_pin, prefetch=_prefetch, in_ram=_in_ram)
    val_draw, n_val = None, 0
    if os.path.isfile(args.val_bin):
        val_draw, n_val = make_data_loader(args.val_bin, args.seq, args.batch_size, args.seed + 1,
                                           pin=_pin, prefetch=_prefetch, in_ram=_in_ram)
    print(f"[veritate_85m] train_bytes={n_train:,}  val_bytes={n_val:,}", flush=True)

    # torch.compile fuses many small ops into bigger CUDA kernels. First step
    # pays ~30-60s compile cost; subsequent steps run a fused graph. We compile
    # AFTER load_state_dict so the saved keys keep their plain names (no
    # `_orig_mod.` prefix). Opt-in via VERITATE_85M_COMPILE=1.
    fwd_model = model
    if os.environ.get("VERITATE_85M_COMPILE", "0") == "1" and device_type == "cuda":
        print("[veritate_85m] torch.compile enabled (mode=default)", flush=True)
        fwd_model = torch.compile(model, mode="default", dynamic=False)

    model.train()
    t0 = time.time()
    last_log = t0
    log_buf_loss, log_buf_l1, log_buf_n = 0.0, 0.0, 0

    for step in range(start_step + 1, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                   schedule=args.lr_schedule,
                   wsd_decay_frac=args.wsd_decay_frac, wsd_decay_kind=args.wsd_decay_kind)
        for g in opt.param_groups:
            g["lr"] = lr

        toks, tgts = train_draw()
        toks = toks.to(device, non_blocking=_pin)
        tgts = tgts.to(device, non_blocking=_pin)

        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                logits, ce, l1_mean = fwd_model(toks, tgts)
        else:
            logits, ce, l1_mean = model(toks, tgts)

        if torch.isnan(ce) or torch.isinf(ce):
            print(f"[veritate_85m] step {step}: NaN/Inf loss — skipping", flush=True)
            continue

        # L1 sparsity penalty on post-ReLU activations. l1_mean is computed
        # inside the model forward so the per-layer .abs().mean() ops run as
        # part of the same autograd graph (no Python-side reduction loop).
        if args.l1_lambda > 0:
            l1 = args.l1_lambda * l1_mean
            loss = ce + l1
            l1_val = float(l1.detach())
        else:
            loss = ce
            l1_val = 0.0

        loss.backward()
        gnorm = None
        if args.grad_clip > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        opt.step()

        log_buf_loss += float(ce.detach())
        log_buf_l1 += l1_val
        log_buf_n += 1

        if step % args.log_every == 0:
            mean_ce = log_buf_loss / max(1, log_buf_n)
            mean_l1 = log_buf_l1 / max(1, log_buf_n)
            now = time.time()
            tok_s = (args.log_every * args.batch_size * args.seq) / max(1e-6, now - last_log)
            save.append_train_row(name, step, "train", mean_ce, lr=lr,
                                  grad_norm=gnorm, tok_per_s=tok_s, wall_s=now - t0,
                                  seed=args.seed)
            print(f"[veritate_85m] step {step:>6} ce={mean_ce:.4f} l1={mean_l1:.4f} "
                  f"lr={lr:.2e} tok/s={tok_s:.0f} t={now-t0:.0f}s", flush=True)
            log_buf_loss, log_buf_l1, log_buf_n = 0.0, 0.0, 0
            last_log = now

        if val_draw is not None and (step % args.eval_every == 0 or step == args.total_steps):
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                vn = 0
                for _ in range(args.eval_iters):
                    vtoks, vtgts = val_draw()
                    vtoks = vtoks.to(device, non_blocking=_pin)
                    vtgts = vtgts.to(device, non_blocking=_pin)
                    if amp_dtype is not None:
                        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                            _, vce, _ = fwd_model(vtoks, vtgts)
                    else:
                        _, vce, _ = fwd_model(vtoks, vtgts)
                    if not (torch.isnan(vce) or torch.isinf(vce)):
                        vloss += float(vce)
                        vn += 1
                vmean = vloss / max(1, vn)
            model.train()
            save.append_train_row(name, step, "val", vmean, lr=lr, wall_s=time.time() - t0,
                                  seed=args.seed)
            print(f"[veritate_85m] step {step:>6} val={vmean:.4f}", flush=True)

        if step % args.ckpt_every == 0 or step == args.total_steps:
            ckpt_args = vars(args).copy()
            # Pull in the model-shape fields the dump suite + config.json expect.
            ckpt_args["vocab"]     = model.vocab
            ckpt_args["hidden"]    = model.hidden
            ckpt_args["layers"]    = model.layers
            ckpt_args["ffn"]       = model.ffn
            ckpt_args["heads"]     = model.heads
            ckpt_args["seq"]       = model.seq
            ckpt_args["n_predict"] = model.n_predict
            ckpt_args.setdefault("description", args.description)
            path = save.save(model, name, step, optimizer=opt, args=ckpt_args)
            print(f"[veritate_85m] checkpoint + hooks: {path}", flush=True)

    print(f"[veritate_85m] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
