# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tool-use SFT trainer. Loads a pretrained base ckpt and continues training
#   on a tool-use SFT corpus (a .bin produced by veritate_mri/tools/jsonl_to_bin.py
#   from Mintaka's W13/W15 tool-use JSONL traces).
# - Cross-model: works with either canonical Veritate or Veritate800M (RoPE +
#   MTP). The base ckpt's state-dict shape decides which class to instantiate.
#   No isinstance checks in the loop (preflight rule 11a) — only at the single
#   construction site.
# - SFT recipe defaults: LR 5e-5 (low, post-pretraining), 5000 steps, WSD
#   schedule with decay_frac=0.1 (so any intermediate ckpt is near-best),
#   no MTP-aux loss (the head still works, just no auxiliary weight). No
#   prompt-vs-response loss masking yet — the corpus is high-signal SFT
#   data; we train on next-byte across the entire trace.
# - Output model name: <base_ckpt_name>_tool_sft_v<N>. Lives in models/<name>/
#   alongside its own train.csv and checkpoints (same layout as other
#   trainer plugins). save.save() emits the standard hook dumps.
# - Uses the platform `veritate_core.plugin` surface (save, paths, model). No
#   direct imports from veritate_mri internals (preflight rule 39).
# plugins/veritate_tool_sft/plugin.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from veritate_core.plugin import save, paths, model as _model_mod

# 800M class is defined in the 800M plugin (not in the platform surface,
# to keep the plugin contract narrow). We import it here ONLY at the
# single construction-site dispatch.
sys.path.insert(0, os.path.join(REPO_ROOT, "plugins", "veritate_800m"))

Veritate         = _model_mod.Veritate
VOCAB_BYTE_LEVEL = _model_mod.VOCAB_BYTE_LEVEL

append_train_row    = save.append_train_row
require_description = save.require_description

# ------------------------------------------------------------------------------------
# Constants

with open(os.path.join(HERE, "manifest.json"), "r") as f:
    MANIFEST = json.load(f)

LR_SCHEDULES    = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS = ("sqrt", "linear", "cosine")

# ------------------------------------------------------------------------------------
# Functions


def parse_args():
    ap = argparse.ArgumentParser(description=MANIFEST.get("description", ""))
    ap.add_argument("--base_ckpt",   type=str, required=True,
                    help="Path to a pretrained .pt to start from.")
    ap.add_argument("--sft_corpus",  type=str, required=True,
                    help="Path to the tool-SFT train.bin (output of jsonl_to_bin).")
    ap.add_argument("--sft_val",     type=str, default="",
                    help="Optional held-out val.bin (same format).")
    ap.add_argument("--device",      type=str, default="auto",
                    choices=("auto", "cpu", "cuda", "mps"),
                    help="Force a device; 'auto' picks the best available.")
    ap.add_argument("--output_name", type=str, default="",
                    help="Output model dir name. If empty, derived as "
                         "<base_stem>_toolsft (base must be a compliant name).")
    ap.add_argument("--description", type=str, default="")
    # Core Plugins inject these for trainers that build their model from
    # scratch. tool_sft resumes from an existing checkpoint so activation is
    # locked to whatever the base model used; declared here only so argparse
    # accepts them when the dashboard sends them.
    ap.add_argument("--activation",  type=str,   default="gelu")
    ap.add_argument("--l1_lambda",   type=float, default=0.0)
    for k, v in MANIFEST.get("defaults", {}).items():
        if k in ("activation", "l1_lambda"):
            continue
        if isinstance(v, bool):
            ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(v))
        elif isinstance(v, int):
            ap.add_argument("--" + k, type=int,   default=v)
        elif isinstance(v, float):
            ap.add_argument("--" + k, type=float, default=v)
        else:
            ap.add_argument("--" + k, type=str,   default=str(v))
    args, _unused = ap.parse_known_args()
    return args


def lr_at(step, total, warmup, base_lr, min_lr, schedule="wsd",
          wsd_decay_frac=0.1, wsd_decay_kind="sqrt"):
    """Same WSD-aware lr_at as the other plugins; kept here to preserve the
    self-contained-plugin invariant (preflight rule 35)."""
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


def pick_device():
    forced = (os.environ.get("VERITATE_DEVICE") or "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        ok = (forced == "cpu") \
             or (forced == "cuda" and torch.cuda.is_available()) \
             or (forced == "mps"  and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if ok: return forced
        print(f"[tool_sft] requested device={forced!r} unavailable; using cpu", flush=True)
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_data_loader(bin_path, total_chunk_len, batch_size, seed):
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    N = len(arr)
    rng = np.random.RandomState(seed)
    if N < total_chunk_len + 2:
        raise ValueError("corpus too small for chunk length: " + str(N) + " < " + str(total_chunk_len + 2))

    def draw():
        starts = rng.randint(0, N - total_chunk_len - 1, size=batch_size, dtype=np.int64)
        toks = np.empty((batch_size, total_chunk_len), dtype=np.int64)
        tgts = np.empty((batch_size, total_chunk_len), dtype=np.int64)
        for b, s in enumerate(starts):
            toks[b] = arr[s:s + total_chunk_len]
            tgts[b] = arr[s + 1:s + 1 + total_chunk_len]
        return torch.from_numpy(toks), torch.from_numpy(tgts)

    return draw, N


def build_model_from_ckpt(ckpt_path, device):
    """Single construction-site dispatch on the base ckpt's shape. The state
    dict tells us which class to instantiate; we never branch elsewhere
    (preflight rule 11a)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError(f"base ckpt has no 'config' field: {ckpt_path}")
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"unrecognized base ckpt format: {ckpt_path}")
    is_800m_style = any(k.startswith("mtp.") for k in sd) or any(k.startswith("rope") for k in sd)
    if is_800m_style:
        from plugin import Veritate800M
        m = Veritate800M(vocab=cfg["vocab"], hidden=cfg["hidden"], layers=cfg["layers"],
                         ffn=cfg["ffn"], heads=cfg["heads"], seq=cfg["seq"],
                         n_predict=cfg.get("n_predict", 4),
                         rope_base=cfg.get("rope_base", 10000.0))
    else:
        m = Veritate(vocab=cfg["vocab"], hidden=cfg["hidden"], layers=cfg["layers"],
                     ffn=cfg["ffn"], heads=cfg["heads"], seq=cfg["seq"])
    m.load_state_dict(sd, strict=False)
    return m.to(device), cfg, ckpt.get("step", 0), is_800m_style


def chunked_step(model, tokens, targets, seq, amp_dtype, *, backward=False,
                 device_type="cuda"):
    """One forward pass + loss. Returns scalar loss (or None if backward=True
    and a NaN was caught). amp_dtype=None disables autocast (CPU path)."""
    def _forward():
        logits = model(tokens)
        if isinstance(logits, tuple):
            logits = logits[0]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
    if backward:
        if amp_dtype is not None:
            with torch.amp.autocast(device_type, dtype=amp_dtype):
                loss = _forward()
        else:
            loss = _forward()
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        loss.backward()
        return float(loss.item())
    with torch.no_grad():
        if amp_dtype is not None:
            with torch.amp.autocast(device_type, dtype=amp_dtype):
                loss = _forward()
        else:
            loss = _forward()
    return float(loss.item())


@torch.no_grad()
def evaluate(model, val_draw, n_iters, seq, amp_dtype, device_type):
    model.eval()
    losses = []
    for _ in range(n_iters):
        toks, tgts = val_draw()
        toks = toks.to(next(model.parameters()).device, non_blocking=True)
        tgts = tgts.to(next(model.parameters()).device, non_blocking=True)
        loss = chunked_step(model, toks, tgts, seq, amp_dtype,
                            backward=False, device_type=device_type)
        if loss is not None:
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else None


def main():
    args = parse_args()
    require_description(args.description)
    if not os.path.isfile(args.base_ckpt):
        raise FileNotFoundError("base_ckpt not found: " + args.base_ckpt)
    if not os.path.isfile(args.sft_corpus):
        raise FileNotFoundError("sft_corpus not found: " + args.sft_corpus)
    if args.lr_schedule not in LR_SCHEDULES:
        raise ValueError("unknown lr_schedule: " + str(args.lr_schedule))
    if args.lr_schedule == "wsd" and args.wsd_decay_kind not in WSD_DECAY_KINDS:
        raise ValueError("unknown wsd_decay_kind: " + str(args.wsd_decay_kind))

    device_type = pick_device() if args.device == "auto" else args.device
    device      = torch.device(device_type)
    amp_dtype   = torch.bfloat16 if args.precision == "bf16" and device_type != "cpu" else None

    # Compose output model name: <base_stem>_toolsft (single-token variant
    # per readers.models.NAME_RE). Caller can override via --output_name when
    # the base ckpt's stem isn't a compliant model name on its own.
    base_stem = os.path.splitext(os.path.basename(args.base_ckpt))[0]
    name = args.output_name or f"{base_stem}_toolsft"
    os.makedirs(paths.model_dir(name), exist_ok=True)

    print(f"[tool_sft] device={device_type} amp={amp_dtype} name={name}", flush=True)
    print(f"[tool_sft] base={args.base_ckpt} sft_corpus={args.sft_corpus}", flush=True)

    model, cfg, base_step, is_800m_style = build_model_from_ckpt(args.base_ckpt, device)
    print(f"[tool_sft] loaded model: {type(model).__name__}  "
          f"hidden={cfg['hidden']}  layers={cfg['layers']}  base_step={base_step}", flush=True)

    # Write the config so the dashboard / dumper can read it
    with open(paths.config_path(name), "w") as f:
        json.dump(cfg, f, indent=2)

    train_draw, n_train = make_data_loader(args.sft_corpus, args.seq, args.batch_size, args.seed)
    val_draw, n_val = None, 0
    if args.sft_val and os.path.isfile(args.sft_val):
        val_draw, n_val = make_data_loader(args.sft_val, args.seq, args.batch_size,
                                            args.seed + 1)
    print(f"[tool_sft] train_bytes={n_train}  val_bytes={n_val}", flush=True)

    # Optimizer + AdamW with WSD schedule
    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.base_lr,
                            betas=(args.beta1, args.beta2),
                            eps=1e-6,
                            weight_decay=args.weight_decay,
                            foreach=True)

    model.train()
    t0 = time.time()
    last_log = t0
    log_buf_loss, log_buf_n = 0.0, 0

    for step in range(1, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                   schedule=args.lr_schedule,
                   wsd_decay_frac=args.wsd_decay_frac, wsd_decay_kind=args.wsd_decay_kind)
        for g in opt.param_groups:
            g["lr"] = lr

        toks, tgts = train_draw()
        toks = toks.to(device, non_blocking=False)
        tgts = tgts.to(device, non_blocking=False)

        opt.zero_grad(set_to_none=True)
        loss = chunked_step(model, toks, tgts, args.seq, amp_dtype,
                            backward=True, device_type=device_type)
        if loss is None:
            # NaN skip — same guard the other trainers use
            continue
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        log_buf_loss += loss
        log_buf_n += 1

        if step % args.log_every == 0:
            mean_loss = log_buf_loss / max(1, log_buf_n)
            now = time.time()
            tok_s = (args.log_every * args.batch_size * args.seq) / max(1e-6, now - last_log)
            append_train_row(name, step, "train", mean_loss, lr=lr,
                             grad_norm=None, wall_s=now - t0)
            print(f"[tool_sft] step {step}  loss={mean_loss:.4f}  lr={lr:.2e}  "
                  f"tok/s={tok_s:.0f}", flush=True)
            log_buf_loss, log_buf_n = 0.0, 0
            last_log = now

        if val_draw is not None and (step % args.eval_every == 0 or step == args.total_steps):
            v = evaluate(model, val_draw, args.eval_iters, args.seq, amp_dtype, device_type)
            if v is not None:
                append_train_row(name, step, "val", v, lr=lr, wall_s=time.time() - t0)
                print(f"[tool_sft] step {step}  val_loss={v:.4f}", flush=True)

        if step % args.ckpt_every == 0 or step == args.total_steps:
            save.save(model, name, step, optimizer=opt, args=vars(args))
            print(f"[tool_sft] saved ckpt at step {step}", flush=True)

    print(f"[tool_sft] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
