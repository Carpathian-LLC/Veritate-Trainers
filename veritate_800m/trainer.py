# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_800m: the flagship Veritate trunk. ~800M params, byte-level
#   (vocab=256), RoPE positional encoding, 4-byte multi-token-prediction (MTP)
#   head. Trained from scratch on FineWeb-Edu for ~16B tokens.
# - Architecture upgrades over canonical Veritate:
#     1. RoPE (no learned pos_emb) — supports length extrapolation past
#        trained seq, enables every 2026-frontier long-context technique.
#     2. MTP head with N=4 — predicts the next 4 bytes per forward, enabling
#        a ~3x decode-speed boost at inference for free.
# - Optimization stack: single-forward (no chunked_step), foreach AdamW,
#   prefetched data loader, bf16 autocast on MPS, NaN-skip guard, bf16-native
#   RMSNorm. Same recipe as the proven 85M overnight run.
# - Hyperparameters tuned for QUALITY (Chinchilla-optimal token budget) over
#   speed. base_lr=3e-4, batch=32, seq=1024, warmup=2000, cosine LR decay.
# plugins/veritate_800m/plugin.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import json
import math
import os
import sys
import threading
import time

# IMPORTANT: this env var must be set BEFORE `import torch`. Otherwise the MPS
# allocator initializes with its default high-water-mark cap (~75% of unified
# memory) and the OOM-kill happens before our Python code runs. Setting it at
# main() time is too late.
if not os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from veritate_core.plugin import save, paths, qat as qat_helpers, multicorpus, bench
from veritate_core.model import RMSNorm, QuantLinear, FFN, VOCAB_BYTE_LEVEL
from veritate_core import qat as _qat


with open(os.path.join(HERE, "manifest.json"), "r", encoding="utf-8") as _f:
    MANIFEST = json.load(_f)


# ------------------------------------------------------------------------------------
# Constants

# Shape table comes from manifest.json's `sizes` block. 800M flagship lives
# there alongside smaller variants; the dashboard's size dropdown and VRAM
# estimator read the same source.
SIZE_PRESETS = {
    k: {field: v[field] for field in ("layers", "hidden", "ffn", "heads")}
    for k, v in (MANIFEST.get("sizes") or {}).items()
}

BASE_CKPT_PREFIX = "step_"
BASE_CKPT_SUFFIX = ".pt"

LR_SCHEDULES = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS = ("sqrt", "linear", "cosine")
PRECISIONS   = ("fp32", "bf16")


# ------------------------------------------------------------------------------------
# RoPE (no parameters)

def build_rope_cache(d_head, max_seq, base=10000.0, device=None, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    t   = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    T = x.size(-2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_t = cos[:T].view(1, 1, T, -1)
    sin_t = sin[:T].view(1, 1, T, -1)
    rx1 = x1 * cos_t - x2 * sin_t
    rx2 = x1 * sin_t + x2 * cos_t
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


# ------------------------------------------------------------------------------------
# Model

class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.h    = heads
        self.d    = hidden // heads
        self.qkv  = QuantLinear(hidden, 3 * hidden, bias=False)
        self.proj = QuantLinear(hidden, hidden,     bias=False)
        self.qat  = False
        self.engine_faithful = False

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos = rope_cos.to(q.dtype)
        sin = rope_sin.to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.qat and self.engine_faithful:
            q = _qat.fake_quant_act(q)
            k = _qat.fake_quant_act(k)
            v = _qat.fake_quant_act(v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.qat and self.engine_faithful:
            out = _qat.fake_quant_act(out)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, hidden, ffn, heads, activation="gelu", capture_l1=False):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = CausalSelfAttentionRoPE(hidden, heads)
        self.n2   = RMSNorm(hidden)
        self.ff   = FFN(hidden, ffn, activation=activation, capture_l1=capture_l1)
        self.qat  = False

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.n1(x), rope_cos, rope_sin)
        if self.qat: x = _qat.fake_quant_act(x)
        x = x + self.ff(self.n2(x))
        if self.qat: x = _qat.fake_quant_act(x)
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
        self.lm_head = lm_head  # tied with tok_emb upstream

    def forward(self, h):
        outs = []
        for i in range(self.n_predict):
            outs.append(self.lm_head(self.norms[i](self.transforms[i](h))))
        return torch.stack(outs, dim=2)  # [B, T, N, vocab]


class Veritate800M(nn.Module):
    """Veritate 800M flagship trunk. RoPE + MTP. Byte-level (vocab=256)."""

    def __init__(self, vocab, hidden, layers, ffn, heads, seq,
                 n_predict=4, rope_base=10000.0,
                 activation="gelu", capture_l1=False):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.vocab      = vocab
        self.hidden     = hidden
        self.layers     = layers
        self.ffn        = ffn
        self.heads      = heads
        self.seq        = seq
        self.n_predict  = n_predict
        self.rope_base  = rope_base
        self.d_head     = hidden // heads
        self.qat        = False
        self.activation = activation
        self.capture_l1 = bool(capture_l1)
        self._mtp_aux_weight = 0.1

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.blocks  = nn.ModuleList([Block(hidden, ffn, heads,
                                            activation=activation,
                                            capture_l1=self.capture_l1)
                                      for _ in range(layers)])
        self.n_out   = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        # NOTE: tying the lm_head weight to tok_emb.weight produced NaN grads
        # on a single embedding row at training time (autograd's accumulation
        # of the lm_head path's grad with the embedding-lookup path's grad
        # through a tied parameter, with N=4 MTP heads reusing the same head,
        # creates a degenerate accumulation case for at least one byte index).
        # Untying — small extra memory cost (256 * hidden = 256K extra params)
        # is worth the numerical stability.
        # self.lm_head.weight = self.tok_emb.weight  # tied (caused NaN grad)

        self.mtp = MTPHead(hidden, n_predict, self.lm_head)

        cos, sin = build_rope_cache(self.d_head, seq, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize parameters that haven't been set already.
        # Skip mtp.transforms[0] (identity-init) and mtp.transforms[1..N-1]
        # (small-random-init in MTPHead.__init__).
        mtp_params = {id(t.weight) for t in self.mtp.transforms}
        for p in self.parameters():
            if p.dim() >= 2 and id(p) not in mtp_params:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def set_qat(self, value):
        return _qat.set_qat(self, value)

    def hook_spec(self):
        return self

    def post_l1_sum(self):
        """Sum of per-block post-activation L1 means captured during the last
        forward. Returns None when capture_l1 is off or no forward has run."""
        if not self.capture_l1:
            return None
        parts = [blk.ff._last_l1 for blk in self.blocks if blk.ff._last_l1 is not None]
        if not parts:
            return None
        return sum(parts)

    def extend_rope(self, new_max_seq):
        device = self.rope_cos.device
        dtype  = self.rope_cos.dtype
        cos, sin = build_rope_cache(self.d_head, new_max_seq, base=self.rope_base,
                                    device=device, dtype=dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def ensure_context(self, T):
        if T > self.rope_cos.size(0):
            self.extend_rope(T)

    def run_blocks(self, x, start_pos=0, exit_after=None):
        n = self.layers if exit_after is None else min(int(exit_after), self.layers)
        for L in range(n):
            x = self.run_block(x, L, start_pos=start_pos)
        return x

    def run_block(self, x, L, start_pos=0):
        return self.blocks[L](x, self.rope_cos, self.rope_sin)

    def project_byte0(self, residual):
        h = self.n_out(residual)
        return self.mtp.lm_head(self.mtp.norms[0](self.mtp.transforms[0](h)))

    def supports_mtp_decode(self):
        return int(self.n_predict) > 1

    def embed(self, tokens, start_pos=0):
        e = self.tok_emb(tokens)
        if self.qat: e = _qat.fake_quant_act(e)
        return e

    def kv_cache_patch_attn(self, attn_mod, cache, get_start_pos):
        def fwd(x, rope_cos, rope_sin):
            B, T, C = x.shape
            h, d = attn_mod.h, attn_mod.d
            qkv = attn_mod.qkv(x).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            start = get_start_pos()
            cos_slice = rope_cos[start:start + T].to(q.dtype)
            sin_slice = rope_sin[start:start + T].to(q.dtype)
            q = apply_rope(q, cos_slice, sin_slice)
            k = apply_rope(k, cos_slice, sin_slice)
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

    def forward(self, tokens, targets=None):
        """Canonical Veritate forward signature: returns (logits, loss).

        - logits: [B, T, vocab] — the byte-0 head's predictions, the same shape
          and meaning every dump function in `veritate_mri/checkpoint_probe.py`
          expects (`logits, _ = model(x)`).
        - loss: total weighted MTP loss (sum over the N heads with `mtp_aux_weight`
          on heads 1..N-1) when `targets` is given; `None` otherwise.

        For the trainer's per-step "byte-0-only" logging metric, call
        `.forward_with_breakdown(tokens, targets)` which returns
        `(logits_b0, total_loss, byte0_loss)`.
        """
        logits_b0, total_loss, _ = self.forward_with_breakdown(tokens, targets)
        return logits_b0, total_loss

    def forward_with_breakdown(self, tokens, targets=None):
        """Trainer-only entry point. Returns (logits_b0, total_loss, byte0_loss)
        so the trainer can log the byte-0 metric (comparable to a canonical
        Veritate's val NLL) while still backproping through total_loss."""
        B, T = tokens.shape
        if T > self.rope_cos.size(0):
            self.extend_rope(T)
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        x = self.n_out(x)
        all_logits = self.mtp(x)            # [B, T, N, vocab]
        logits_b0  = all_logits[:, :, 0, :] # [B, T, vocab]

        total_loss = None
        byte0_loss = None
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
                tgt_i    = head_targets[:, :, i].reshape(-1)
                losses.append(F.cross_entropy(logits_i, tgt_i, ignore_index=-1))
            byte0_loss = losses[0]
            total_loss = sum(weights[i] * losses[i] for i in range(self.n_predict))

        return logits_b0, total_loss, byte0_loss


# ------------------------------------------------------------------------------------
# Trainer plumbing (matching the proven optimization stack)

def _truthy(s):
    if isinstance(s, bool): return s
    return str(s).strip().lower() in ("1", "true", "yes", "on")


_RESERVED_FLAGS = {"name", "corpus", "description", "resume", "activation", "l1_lambda"}


def parse_args():
    ap = argparse.ArgumentParser(description=MANIFEST.get("description", ""))
    ap.add_argument("--name",        type=str, default="")
    ap.add_argument("--corpus",      type=str, default="")
    ap.add_argument("--description", type=str, default="")
    ap.add_argument("--resume",      type=str, default="")
    ap.add_argument("--bench",       action="store_true")
    # Core Plugins inject these; declare unconditionally so argparse accepts them.
    ap.add_argument("--activation",  type=str,   default="gelu")
    ap.add_argument("--l1_lambda",   type=float, default=0.0)
    for k, v in MANIFEST.get("defaults", {}).items():
        # Skip flags that are already declared above (manifest may include
        # `corpus` so the dashboard renders a corpus picker; the actual flag
        # comes from the hard-coded line above).
        if k in _RESERVED_FLAGS:
            continue
        if isinstance(v, bool):
            # The dashboard form passes booleans as bare flags (no value):
            #   `--use_act_ckpt` means True, omitting the flag means False.
            # BooleanOptionalAction handles both `--use_act_ckpt` and
            # `--no-use_act_ckpt`, and respects the manifest default when the
            # flag isn't passed at all.
            ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(v))
        elif isinstance(v, int):
            ap.add_argument("--" + k, type=int,   default=v)
        elif isinstance(v, float):
            ap.add_argument("--" + k, type=float, default=v)
        else:
            ap.add_argument("--" + k, type=str,   default=str(v))
    # The dashboard form renders the FULL TRAINER_SCHEMA for every plugin, so
    # flags this trainer does not implement (quant_mode for non-MoE, etc.)
    # arrive on argv and would otherwise crash parsing. Drop them silently.
    args, _unused = ap.parse_known_args()
    return args


def latest_checkpoint_step(name):
    ckpt_dir = paths.checkpoints_dir(name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("no checkpoints dir for: " + name)
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(BASE_CKPT_PREFIX) and fn.endswith(BASE_CKPT_SUFFIX):
            try:
                steps.append(int(fn[len(BASE_CKPT_PREFIX):-len(BASE_CKPT_SUFFIX)]))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError("no step_*.pt under: " + ckpt_dir)
    return max(steps)


def apply_resume_overrides(args, argv):
    cfg_path = paths.config_path(args.resume)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError("no config.json for resume target: " + args.resume)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ta = cfg.get("training_args") or {}
    for k, v in ta.items():
        if not hasattr(args, k):
            continue
        flag = "--" + k
        if any(a == flag or a.startswith(flag + "=") for a in argv):
            continue
        cur = getattr(args, k)
        if isinstance(cur, bool) and not isinstance(v, bool):
            continue
        try:
            setattr(args, k, type(cur)(v) if cur is not None else v)
        except (TypeError, ValueError):
            setattr(args, k, v)


def lr_at(step, total, warmup, base_lr, min_lr, schedule="cosine",
          wsd_decay_frac=0.1, wsd_decay_kind="sqrt"):
    """LR schedule. Three-arm `wsd` (Warmup-Stable-Decay) keeps base_lr flat
    after warmup until the last `wsd_decay_frac` of training, then decays
    to min_lr. wsd_decay_kind picks the decay shape: sqrt (1-√q, the
    recipe in the WSD paper), linear, or cosine. Continued-pretrain and
    runs of unknown length benefit from WSD because the stable arm makes
    intermediate checkpoints near-optimal; cosine forces a commit to
    `total` up front."""
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
        q = min(max(q, 0.0), 1.0)
        if wsd_decay_kind == "linear":
            shape = 1.0 - q
        elif wsd_decay_kind == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * q))
        else:
            shape = 1.0 - math.sqrt(q)
        return min_lr + (base_lr - min_lr) * shape
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


class Prefetcher:
    def __init__(self, draw_fn):
        self._draw = draw_fn
        self._cv   = threading.Condition()
        self._next = None
        self._die  = False
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while True:
            with self._cv:
                while self._next is not None and not self._die:
                    self._cv.wait()
                if self._die: return
            b = self._draw()
            with self._cv:
                self._next = b
                self._cv.notify_all()
    def next(self):
        with self._cv:
            while self._next is None:
                self._cv.wait()
            b = self._next
            self._next = None
            self._cv.notify_all()
        return b
    def close(self):
        with self._cv:
            self._die = True
            self._cv.notify_all()


def make_data_loader(bin_path, batch_size, seq, seed):
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    N = len(arr)
    rng = np.random.RandomState(seed)
    if N < seq + 2:
        raise ValueError("corpus too small for seq: " + str(N) + " < " + str(seq + 2))

    def draw():
        starts = rng.randint(0, N - seq - 1, size=batch_size, dtype=np.int64)
        toks = np.empty((batch_size, seq), dtype=np.int64)
        tgts = np.empty((batch_size, seq), dtype=np.int64)
        for b, s in enumerate(starts):
            toks[b] = arr[s:s + seq]
            tgts[b] = arr[s + 1:s + 1 + seq]
        # CRITICAL: `torch.from_numpy` shares memory with the numpy array.
        # When this function is called from a Prefetcher worker thread, the
        # next call's `np.empty` can reuse the SAME pooled allocation,
        # silently overwriting the tensor while the main thread is still
        # reading it -> garbage indices -> embedding gather pulls from
        # nonsense rows -> NaN grad on tok_emb.weight. `torch.tensor(np)`
        # forces a copy.
        return torch.tensor(toks), torch.tensor(tgts)

    return draw, N


def pick_device():
    forced = (os.environ.get("VERITATE_DEVICE") or "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        ok = (forced == "cpu") \
             or (forced == "cuda" and torch.cuda.is_available()) \
             or (forced == "mps"  and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if ok: return forced
        print(f"[veritate_800m] requested device={forced!r} unavailable; using cpu", flush=True)
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_config(name, args, base_cfg, n_params, corpus_hash):
    cfg_path = paths.config_path(name)
    os.makedirs(paths.model_dir(name), exist_ok=True)
    ta = vars(args).copy()
    if corpus_hash:
        ta["corpus_sha256"] = corpus_hash.get("train_sha256")
        ta["corpus_bytes"]  = corpus_hash.get("train_bytes")
    shape = dict(base_cfg)
    shape["seq"]       = args.seq
    shape["vocab"]     = VOCAB_BYTE_LEVEL
    shape["n_predict"] = args.n_predict
    shape["rope_base"] = args.rope_base
    qat_on = bool(getattr(args, "qat_enabled", False))
    cfg = {
        "name": name,
        "description": args.description,
        "kind": "trainer",
        "plugin": "veritate_800m",
        "vocab": VOCAB_BYTE_LEVEL,
        "shape": shape,
        "training":  ("qat" if qat_on else ""),
        "qat_source": (args.resume if (qat_on and args.resume) else ""),
        "training_args": ta,
        "n_params_total": n_params,
        "wrote_at": int(time.time()),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_resume_state(model, name, step, device):
    ckpt = torch.load(paths.checkpoint_path(name, step), map_location=device, weights_only=False)
    sd = ckpt["model"]
    expected = set(model.state_dict().keys())
    incoming = set(sd.keys())
    missing  = expected - incoming
    extra    = incoming - expected
    SHAPE_HINT = ("blocks.", "tok_emb", "lm_head", "mtp.")
    shape_missing = [k for k in missing if any(s in k for s in SHAPE_HINT)]
    shape_extra   = [k for k in extra   if any(s in k for s in SHAPE_HINT)]
    if shape_missing or shape_extra:
        raise RuntimeError(
            "checkpoint shape mismatch on resume.\n"
            "  missing in checkpoint: " + str(sorted(shape_missing)[:6]) + "\n"
            "  extra in checkpoint:   " + str(sorted(shape_extra)[:6])
        )
    model.load_state_dict(sd, strict=False)
    return ckpt.get("optimizer")


@torch.no_grad()
def evaluate(model, val_draw, n_iters, amp_dtype, device):
    model.eval()
    losses = []
    for _ in range(n_iters):
        toks, tgts = val_draw()
        toks = toks.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            _, _, b0 = model.forward_with_breakdown(toks, targets=tgts)
        if b0 is not None:
            losses.append(b0.detach())
    model.train()
    if not losses:
        return None
    return float(torch.stack(losses).mean().item())


# ------------------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    resume_mode = bool(args.resume)
    qat_enabled = bool(getattr(args, "qat_enabled", False))
    qat_source  = args.resume if (resume_mode and qat_enabled) else None
    if resume_mode:
        apply_resume_overrides(args, sys.argv)
        if qat_enabled:
            args.qat_enabled = True
    if not args.bench:
        save.require_description(args.description)

    if args.size not in SIZE_PRESETS:
        raise ValueError("unknown size: " + str(args.size) + " (valid: " + ", ".join(SIZE_PRESETS) + ")")
    if args.precision not in PRECISIONS:
        raise ValueError("unknown precision: " + str(args.precision))
    if args.lr_schedule not in LR_SCHEDULES:
        raise ValueError("unknown lr_schedule: " + str(args.lr_schedule))
    if args.lr_schedule == "wsd":
        kind = getattr(args, "wsd_decay_kind", "sqrt")
        if kind not in WSD_DECAY_KINDS:
            raise ValueError("unknown wsd_decay_kind: " + str(kind)
                             + " (valid: " + ", ".join(WSD_DECAY_KINDS) + ")")
        frac = float(getattr(args, "wsd_decay_frac", 0.1))
        if not (0.0 < frac <= 1.0):
            raise ValueError("wsd_decay_frac must be in (0, 1], got " + str(frac))

    if qat_source is not None:
        name = qat_source + "_qat"
        print("QAT continue: source=" + qat_source + " new model=" + name, flush=True)
    elif resume_mode:
        name = args.resume
    elif getattr(args, "name", "").strip():
        slug = save.compose_name(args.name, args.size)
        name = slug + "_qat" if qat_enabled else slug
    else:
        v = args.version
        if v.endswith("_qat"):
            v = v[:-4]
        elif v.endswith("qat"):
            v = v[:-3]
        version_tag = (v + "_qat") if qat_enabled else v
        name = save.compose_name(args.corpus, args.size, args.precision, version_tag)
    print("model name: " + name, flush=True)

    _corpus_mix = None
    val_path    = None
    if not args.bench:
        _corpus_mix = multicorpus.resolve_and_weight(args.corpus, save.resolve_corpus)
        val_path    = _corpus_mix[0][1]
        print("corpus mix:   " + multicorpus.format_mix_summary(_corpus_mix), flush=True)
        if val_path:
            print("corpus val:   " + val_path, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(min(8, max(2, (os.cpu_count() or 4) // 4)))
    device = pick_device()
    print(f"device: {device}", flush=True)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None

    shape = SIZE_PRESETS[args.size]
    activation = getattr(args, "activation", "gelu") or "gelu"
    l1_lambda  = float(getattr(args, "l1_lambda", 0.0) or 0.0)
    model = Veritate800M(
        vocab=VOCAB_BYTE_LEVEL,
        hidden=shape["hidden"], layers=shape["layers"],
        ffn=shape["ffn"], heads=shape["heads"], seq=args.seq,
        n_predict=args.n_predict, rope_base=args.rope_base,
        activation=activation, capture_l1=(l1_lambda > 0.0),
    )
    model._mtp_aux_weight = float(args.mtp_aux_weight)
    print(f"activation: {activation}  l1_lambda: {l1_lambda}", flush=True)

    if qat_enabled:
        qat_helpers.set_qat(model, True)
        print("QAT: enabled", flush=True)

    if args.use_act_ckpt:
        print("activation checkpointing: ENABLED", flush=True)
        for blk in model.blocks:
            orig = blk.forward
            blk.forward = (lambda fwd: lambda x, c, s: torch.utils.checkpoint.checkpoint(fwd, x, c, s, use_reentrant=False))(orig)

    # Pre-flight: MPS's attention path uses int32 for tensor element offsets.
    # If B * heads * seq * seq > INT_MAX (~2.15e9), backward fails 42 seconds in
    # with "MPSGraph does not support tensor dims larger than INT_MAX". Catch
    # it now with a clear message instead of mid-training.
    INT_MAX = 2_147_483_647
    attn_scratch = int(args.batch_size) * int(shape["heads"]) * int(args.seq) * int(args.seq)
    if device == "mps" and attn_scratch > INT_MAX:
        max_b = INT_MAX // (int(shape["heads"]) * int(args.seq) * int(args.seq))
        raise ValueError(
            f"MPS attention-scratch dims overflow int32: "
            f"batch ({args.batch_size}) × heads ({shape['heads']}) × seq ({args.seq})^2 "
            f"= {attn_scratch:,} > INT_MAX ({INT_MAX:,}).\n"
            f"  Reduce one of: batch_size <= {max_b}, seq, or shape['heads']. "
            f"This is an MPSGraph limitation, not a plugin bug. "
            f"Recommended for this shape: batch_size = {max_b}."
        )

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("device: " + device + "  precision: " + args.precision, flush=True)
    print("params: " + str(n_params) + f" ({n_params/1e6:.1f}M)", flush=True)
    print("shape:  hidden=" + str(shape["hidden"]) + " layers=" + str(shape["layers"])
          + " ffn=" + str(shape["ffn"]) + " heads=" + str(shape["heads"])
          + " seq=" + str(args.seq) + " n_predict=" + str(args.n_predict), flush=True)

    if args.bench:
        result = bench.run(model, device, args.seq, VOCAB_BYTE_LEVEL,
                           on_progress=lambda s: print("bench: " + s, flush=True))
        print("BENCH_RESULT " + json.dumps(result), flush=True)
        return

    resume_step = 0
    resume_opt_state = None
    if qat_source is not None:
        src_step = latest_checkpoint_step(qat_source)
        print("QAT load: " + qat_source + " step " + str(src_step) + "  -> new model " + name, flush=True)
        load_resume_state(model, qat_source, src_step, device)
        write_config(name, args, shape, n_params, corpus_hash=None)
        print("wrote: " + paths.config_path(name), flush=True)
    elif resume_mode:
        resume_step = latest_checkpoint_step(name)
        print("resume: " + name + "  from step " + str(resume_step), flush=True)
        resume_opt_state = load_resume_state(model, name, resume_step, device)
    else:
        print("hashing corpus (one-time, may take a few minutes for 200GB)...", flush=True)
        corpus_hash = save.hash_corpus(args.corpus)
        print("corpus sha256: " + corpus_hash.get("train_sha256", "?")[:16] + "...  bytes=" + str(corpus_hash.get("train_bytes")), flush=True)
        write_config(name, args, shape, n_params, corpus_hash)
        print("wrote: " + paths.config_path(name), flush=True)

    train_draw, train_n = multicorpus.make_mixed_loader(_corpus_mix, args.batch_size, args.seq, args.seed)
    val_draw = None
    if val_path:
        val_draw, _ = make_data_loader(val_path, args.batch_size, args.seq, args.seed + 1)
    print("train corpus bytes: " + str(train_n) + "  per-step tokens: " + str(args.batch_size * args.seq), flush=True)

    train_pf = Prefetcher(train_draw)

    opt_kwargs = dict(
        lr=args.base_lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=1e-6,
    )
    if device == "cuda":
        opt_kwargs["fused"] = True
    else:
        opt_kwargs["foreach"] = True
    opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    if resume_opt_state is not None:
        try:
            opt.load_state_dict(resume_opt_state)
            print("optimizer state restored", flush=True)
        except Exception as e:
            print("optimizer state restore skipped: " + str(e), flush=True)

    t0 = time.time()
    last_log = t0
    last_log_step = resume_step
    start_step = resume_step + 1
    skipped = 0
    try:
        for step in range(start_step, args.total_steps + 1):
            lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                       schedule=args.lr_schedule,
                       wsd_decay_frac=getattr(args, "wsd_decay_frac", 0.1),
                       wsd_decay_kind=getattr(args, "wsd_decay_kind", "sqrt"))
            for g in opt.param_groups:
                g["lr"] = lr

            toks, tgts = train_pf.next()
            # NOTE: non_blocking=False is REQUIRED here because the Prefetcher
            # is a separate thread that immediately reuses its source buffer
            # for the next batch via numpy's allocator pool. With non_blocking
            # the CPU->MPS copy is async and the source's first ~16 bytes get
            # corrupted by the prefetcher's next write before the H2D finishes,
            # producing garbage token indices -> NaN grad on tok_emb.weight.
            toks = toks.to(device, non_blocking=False)
            tgts = tgts.to(device, non_blocking=False)

            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _, total_loss, byte0_loss = model.forward_with_breakdown(toks, targets=tgts)
            if not torch.isfinite(total_loss):
                skipped += 1
                continue
            if l1_lambda > 0.0 and model.capture_l1:
                l1 = model.post_l1_sum()
                if l1 is not None:
                    total_loss = total_loss + l1_lambda * l1
            total_loss.backward()
            nan_grad = False
            for _p in model.parameters():
                if _p.grad is not None and not torch.isfinite(_p.grad).all():
                    nan_grad = True
                    break
            if nan_grad:
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            gn_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step % args.log_every == 0 or step == 1:
                now = time.time()
                elapsed = now - t0
                window_s = max(1e-6, now - last_log)
                window_steps = step - last_log_step
                tok_per_s = window_steps * args.batch_size * args.seq / window_s
                tot_v  = float(total_loss.detach().item())
                b0_v   = float(byte0_loss.detach().item())
                gn_v   = float(gn_t.detach().item())
                print("step " + str(step) + "  total " + format(tot_v, ".4f")
                      + "  byte0 " + format(b0_v, ".4f")
                      + "  lr " + format(lr, ".2e")
                      + "  gn " + format(gn_v, ".3f")
                      + "  tok/s " + format(tok_per_s, ".0f")
                      + "  skip " + str(skipped)
                      + "  elapsed " + format(elapsed, ".0f") + "s", flush=True)
                save.append_train_row(name, step, "train", b0_v,
                                      lr=lr, grad_norm=gn_v,
                                      tok_per_s=tok_per_s, wall_s=elapsed, seed=args.seed)
                last_log = now
                last_log_step = step

            if val_draw is not None and step % args.eval_every == 0:
                v = evaluate(model, val_draw, args.eval_iters, amp_dtype, device)
                if v is not None:
                    print("step " + str(step) + "  val_loss " + format(v, ".4f"), flush=True)
                    save.append_train_row(name, step, "val", v, lr=lr,
                                          wall_s=time.time() - t0, seed=args.seed)

            if step % args.ckpt_every == 0 or step == args.total_steps:
                ckpt_args = vars(args).copy()
                ckpt_args["vocab"]     = model.vocab
                ckpt_args["hidden"]    = model.hidden
                ckpt_args["layers"]    = model.layers
                ckpt_args["ffn"]       = model.ffn
                ckpt_args["heads"]     = model.heads
                ckpt_args["seq"]       = model.seq
                ckpt_args["n_predict"] = model.n_predict
                ckpt_args["rope_base"] = model.rope_base
                ckpt_args.setdefault("description", args.description)
                ckpt_path = save.save(model, name, step, optimizer=opt, args=ckpt_args)
                print("checkpoint + hooks: " + ckpt_path, flush=True)
    finally:
        train_pf.close()

    print("done.", flush=True)


if __name__ == "__main__":
    main()
