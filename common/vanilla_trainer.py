# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Shared vanilla-Veritate trainer entry point. Used by every plugin that
#   trains a canonical model.Veritate trunk from scratch (10m, 80m, 200m,
#   400m, 1b3, 3b, 13b, 50b). One module owns the training loop, the LR
#   schedule, the data loader, the chunked forward, the resume logic, the
#   QAT and 8-bit-Adam plumbing, and the config.json bootstrap.
# - The per-plugin file is a thin shim that calls run(plugin_id, here). All
#   shape, LR, batch, schedule values come from the calling plugin's
#   manifest.json. Nothing in this module is keyed by plugin id beyond
#   stamping the id into config.json so the dashboard can group runs.
# - Manifest's `sizes` block is the single source of truth for shape data.
#   Adding a new size means adding a manifest entry, not touching this code.
# plugins/common/vanilla_trainer.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from veritate_core.plugin import save, paths, model as _model_mod, qat as qat_helpers, multicorpus, bench

Veritate         = _model_mod.Veritate
VOCAB_BYTE_LEVEL = _model_mod.VOCAB_BYTE_LEVEL

append_train_row    = save.append_train_row
compose_name        = save.compose_name
hash_corpus         = save.hash_corpus
require_description = save.require_description
resolve_corpus      = save.resolve_corpus


# ------------------------------------------------------------------------------------
# Constants

SHAPE_FIELDS    = ("layers", "hidden", "ffn", "heads")
BASE_CKPT_PREFIX = "step_"
BASE_CKPT_SUFFIX = ".pt"

LR_SCHEDULES    = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS = ("sqrt", "linear", "cosine")
PRECISIONS      = ("fp32", "bf16")


# ------------------------------------------------------------------------------------
# Functions

def _load_manifest(here):
    with open(os.path.join(here, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _size_presets(manifest):
    return {
        k: {field: v[field] for field in SHAPE_FIELDS}
        for k, v in (manifest.get("sizes") or {}).items()
    }


RESERVED_STRING_FLAGS = ("name", "corpus", "description", "resume")
# Reserved bool flags that the dashboard sends for every plugin regardless of
# whether the manifest opts in. Declared unconditionally so a user toggling
# them on the form is always honored, even on plugins whose manifest defaults
# don't list them. Manifest values override the defaults below.
RESERVED_BOOL_FLAGS = {
    "use_act_ckpt":  False,
    "use_8bit_adam": False,
    "qat_enabled":   False,
}

# Knobs the dashboard's Core Plugins section may inject regardless of whether
# the trainer's manifest opts in. Declared unconditionally so a user toggling
# a Core Plugin always has its args parsed cleanly. Manifest values override.
RESERVED_STR_FLAGS = {
    "activation": "gelu",
}
RESERVED_FLOAT_FLAGS = {
    "l1_lambda": 0.0,
}


def parse_args(manifest):
    ap = argparse.ArgumentParser(description=manifest.get("description", ""))
    ap.add_argument("--bench", action="store_true")
    for k in RESERVED_STRING_FLAGS:
        ap.add_argument("--" + k, type=str, default="")
    defaults = manifest.get("defaults", {}) or {}
    for k, default_val in RESERVED_BOOL_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(manifest_val))
    for k, default_val in RESERVED_STR_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, type=str, default=str(manifest_val))
    for k, default_val in RESERVED_FLOAT_FLAGS.items():
        manifest_val = defaults.get(k, default_val)
        ap.add_argument("--" + k, type=float, default=float(manifest_val))
    for k, v in defaults.items():
        if k in RESERVED_STRING_FLAGS or k in RESERVED_BOOL_FLAGS:
            continue
        if k in RESERVED_STR_FLAGS or k in RESERVED_FLOAT_FLAGS:
            continue
        if isinstance(v, bool):
            ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(v))
        elif isinstance(v, int):
            ap.add_argument("--" + k, type=int,   default=v)
        elif isinstance(v, float):
            ap.add_argument("--" + k, type=float, default=v)
        else:
            ap.add_argument("--" + k, type=str,   default=str(v))
    # Dashboard renders the full TRAINER_SCHEMA for every plugin, so flags the
    # plugin does not implement (quant_mode for non-MoE, freeze_base, etc.)
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
        user_set = any(a == flag or a.startswith(flag + "=") for a in argv)
        if user_set:
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


DEVICE_ENV = "VERITATE_DEVICE"


def _device_available(name):
    if name == "cpu":  return True
    if name == "cuda": return torch.cuda.is_available()
    if name == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return False


def pick_device():
    """Trainer-side device selection. The platform may force a specific device
    via VERITATE_DEVICE=cpu/cuda/mps; we obey it (falling back to cpu if the
    requested device isn't actually available). Otherwise pick best-available.
    The trainer makes no host-architecture assumptions — that's the platform's
    job, communicated through the env var."""
    forced = (os.environ.get(DEVICE_ENV) or "").strip().lower()
    if forced and forced != "auto":
        if forced in ("cpu", "cuda", "mps"):
            if _device_available(forced):
                return forced
            print(f"[vanilla_trainer] requested device={forced!r} unavailable; using cpu", flush=True)
            return "cpu"
        print(f"[vanilla_trainer] ignoring unknown {DEVICE_ENV}={forced!r}", flush=True)
    if torch.cuda.is_available():
        return "cuda"
    if _device_available("mps"):
        return "mps"
    return "cpu"


def chunked_step(model, tokens, targets, seq, amp_dtype, *, backward=False, bptt_window=1,
                 device_type="cuda", l1_lambda=0.0):
    B, total_len = tokens.shape
    n_chunks = max(1, total_len // seq)
    K = max(1, int(bptt_window))
    loss_sum = 0.0
    n_valid  = 0
    window_losses = []
    capture_l1 = bool(getattr(model, "capture_l1", False)) and l1_lambda > 0.0
    for cstart in range(0, total_len, seq):
        cend = min(cstart + seq, total_len)
        ct = tokens[:, cstart:cend]
        cg = targets[:, cstart:cend]
        if ct.size(1) < 2:
            break
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            _, loss = model(ct, targets=cg)
        if loss is None or not torch.isfinite(loss):
            continue
        if capture_l1:
            l1 = model.post_l1_sum()
            if l1 is not None:
                loss = loss + l1_lambda * l1
        loss_sum += float(loss.detach().item())
        n_valid  += 1
        if backward:
            window_losses.append(loss)
            window_full = len(window_losses) >= K
            last_chunk  = (cstart + seq) >= total_len
            if window_full or last_chunk:
                (torch.stack(window_losses).sum() / n_chunks).backward()
                window_losses = []
    if n_valid == 0:
        return None
    return loss_sum / n_valid


def write_config(name, args, base_cfg, n_params, corpus_hash, plugin_id):
    cfg_path = paths.config_path(name)
    os.makedirs(paths.model_dir(name), exist_ok=True)
    ta = vars(args).copy()
    if corpus_hash:
        ta["corpus_sha256"] = corpus_hash.get("train_sha256")
        ta["corpus_bytes"]  = corpus_hash.get("train_bytes")
    shape = dict(base_cfg)
    shape["seq"]   = args.seq
    shape["vocab"] = VOCAB_BYTE_LEVEL
    qat_on = bool(getattr(args, "qat_enabled", False))
    cfg = {
        "name": name,
        "description": args.description,
        "kind": "trainer",
        "plugin": plugin_id,
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
    if any(k.startswith("base.") for k in sd):
        new_sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
        model.load_state_dict(new_sd, strict=False)
    else:
        model.load_state_dict(sd, strict=False)
    return ckpt.get("optimizer")


@torch.no_grad()
def evaluate(model, val_draw, n_iters, seq, amp_dtype, bptt_window, device_type="cuda"):
    model.eval()
    losses = []
    for _ in range(n_iters):
        toks, tgts = val_draw()
        toks = toks.to(next(model.parameters()).device, non_blocking=True)
        tgts = tgts.to(next(model.parameters()).device, non_blocking=True)
        loss = chunked_step(model, toks, tgts, seq, amp_dtype,
                            bptt_window=bptt_window, device_type=device_type)
        if loss is not None:
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else None


def build_optimizer(params, args, device):
    use_8bit = bool(getattr(args, "use_8bit_adam", False))
    if use_8bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(
            params,
            lr=args.base_lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), eps=1e-6,
        )
    return torch.optim.AdamW(
        params,
        lr=args.base_lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=1e-6,
        fused=(device == "cuda"),
    )


def run(plugin_id, here):
    manifest = _load_manifest(here)
    size_presets = _size_presets(manifest)
    if not size_presets:
        raise ValueError("manifest.sizes missing or empty for plugin: " + plugin_id)

    args = parse_args(manifest)
    resume_mode = bool(args.resume)
    qat_enabled = bool(getattr(args, "qat_enabled", False))
    qat_source  = args.resume if (resume_mode and qat_enabled) else None
    if resume_mode:
        apply_resume_overrides(args, sys.argv)
        if qat_enabled:
            args.qat_enabled = True
    if not args.bench:
        require_description(args.description)

    if args.size not in size_presets:
        raise ValueError("unknown size: " + str(args.size) + " (valid: " + ", ".join(size_presets) + ")")
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
        # Dashboard scratch path: user-supplied slug, size suffix appended by
        # save.compose_name (matches the native trainer convention).
        slug = compose_name(args.name, args.size)
        name = slug + "_qat" if qat_enabled else slug
    else:
        v = args.version
        if v.endswith("_qat"):
            v = v[:-4]
        elif v.endswith("qat"):
            v = v[:-3]
        version_tag = (v + "_qat") if qat_enabled else v
        name = compose_name(args.corpus, args.size, args.precision, version_tag)
    print("model name: " + name, flush=True)

    _corpus_mix = None
    val_path    = None
    if not args.bench:
        _corpus_mix = multicorpus.resolve_and_weight(args.corpus, resolve_corpus)
        val_path    = _corpus_mix[0][1]
        print("corpus mix:   " + multicorpus.format_mix_summary(_corpus_mix), flush=True)
        if val_path:
            print("corpus val:   " + val_path, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print("device: " + device, flush=True)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None

    shape = size_presets[args.size]
    activation = getattr(args, "activation", "gelu") or "gelu"
    l1_lambda  = float(getattr(args, "l1_lambda", 0.0) or 0.0)
    veritate_model = Veritate(
        vocab=VOCAB_BYTE_LEVEL,
        hidden=shape["hidden"], layers=shape["layers"],
        ffn=shape["ffn"], heads=shape["heads"], seq=args.seq,
        activation=activation,
        capture_l1=(l1_lambda > 0.0),
    )
    print(f"activation: {activation}  l1_lambda: {l1_lambda}", flush=True)
    if qat_enabled:
        qat_helpers.set_qat(veritate_model, True)
        print("QAT: enabled (fake-quant matmuls + embeddings + RMSNorm + residual adds)", flush=True)

    if getattr(args, "use_act_ckpt", False):
        print("activation checkpointing: ENABLED", flush=True)
        for blk in veritate_model.blocks:
            blk.forward = (lambda fwd: lambda x: torch.utils.checkpoint.checkpoint(fwd, x, use_reentrant=False))(blk.forward)

    veritate_model.to(device)
    n_params = sum(p.numel() for p in veritate_model.parameters())
    print("device: " + device + "  precision: " + args.precision, flush=True)
    print("params: " + str(n_params), flush=True)
    print("shape:  hidden=" + str(shape["hidden"]) + " layers=" + str(shape["layers"])
          + " ffn=" + str(shape["ffn"]) + " heads=" + str(shape["heads"])
          + " seq=" + str(args.seq), flush=True)

    if args.bench:
        result = bench.run(veritate_model, device, args.seq, VOCAB_BYTE_LEVEL,
                           on_progress=lambda s: print("bench: " + s, flush=True))
        print("BENCH_RESULT " + json.dumps(result), flush=True)
        return

    resume_step = 0
    resume_opt_state = None
    if qat_source is not None:
        src_step = latest_checkpoint_step(qat_source)
        print("QAT load: " + qat_source + " step " + str(src_step) + "  -> new model " + name, flush=True)
        load_resume_state(veritate_model, qat_source, src_step, device)
        write_config(name, args, shape, n_params, corpus_hash=None, plugin_id=plugin_id)
        print("wrote: " + paths.config_path(name), flush=True)
    elif resume_mode:
        resume_step = latest_checkpoint_step(name)
        print("resume: " + name + "  from step " + str(resume_step), flush=True)
        resume_opt_state = load_resume_state(veritate_model, name, resume_step, device)
    else:
        print("hashing corpus (one-time, ~5-10s for 2GB)...", flush=True)
        corpus_hash = hash_corpus(args.corpus)
        print("corpus sha256: " + corpus_hash.get("train_sha256", "?")[:16] + "...  bytes=" + str(corpus_hash.get("train_bytes")), flush=True)
        write_config(name, args, shape, n_params, corpus_hash, plugin_id=plugin_id)
        print("wrote: " + paths.config_path(name), flush=True)

    total_chunk_len = args.seq * args.n_chunks
    train_draw, train_n = multicorpus.make_mixed_loader(_corpus_mix, args.batch_size, total_chunk_len, args.seed)
    val_draw = None
    if val_path:
        val_draw, _ = make_data_loader(val_path, total_chunk_len, args.batch_size, args.seed + 1)
    print("train corpus bytes: " + str(train_n) + "  per-step chunk: " + str(total_chunk_len) + "  batch: " + str(args.batch_size), flush=True)

    opt = build_optimizer(veritate_model.parameters(), args, device)
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
    for step in range(start_step, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                   schedule=args.lr_schedule,
                   wsd_decay_frac=getattr(args, "wsd_decay_frac", 0.1),
                   wsd_decay_kind=getattr(args, "wsd_decay_kind", "sqrt"))
        for g in opt.param_groups:
            g["lr"] = lr

        toks, tgts = train_draw()
        toks = toks.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        veritate_model.train()
        opt.zero_grad(set_to_none=True)
        loss = chunked_step(veritate_model, toks, tgts, args.seq, amp_dtype,
                            backward=True, bptt_window=args.bptt_window,
                            device_type=device, l1_lambda=l1_lambda)
        if loss is None:
            continue
        gn = torch.nn.utils.clip_grad_norm_(veritate_model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            elapsed = now - t0
            window_s = max(1e-6, now - last_log)
            window_steps = step - last_log_step
            tok_per_s = window_steps * args.batch_size * total_chunk_len / window_s
            print("step " + str(step) + "  loss " + format(loss, ".4f") + "  lr " + format(lr, ".2e")
                  + "  gn " + format(float(gn), ".3f") + "  tok/s " + format(tok_per_s, ".0f")
                  + "  elapsed " + format(elapsed, ".0f") + "s", flush=True)
            append_train_row(name, step, "train", float(loss),
                             lr=lr, grad_norm=float(gn),
                             tok_per_s=tok_per_s, wall_s=elapsed, seed=args.seed)
            last_log = now
            last_log_step = step

        if val_draw is not None and step % args.eval_every == 0:
            v = evaluate(veritate_model, val_draw, args.eval_iters, args.seq, amp_dtype,
                         args.bptt_window, device_type=device)
            if v is not None:
                print("step " + str(step) + "  val_loss " + format(v, ".4f"), flush=True)
                append_train_row(name, step, "val", v, lr=lr,
                                 wall_s=time.time() - t0, seed=args.seed)

        if step % args.ckpt_every == 0 or step == args.total_steps:
            ckpt_args = vars(args).copy()
            ckpt_args["vocab"]  = veritate_model.vocab
            ckpt_args["hidden"] = veritate_model.hidden
            ckpt_args["layers"] = veritate_model.layers
            ckpt_args["ffn"]    = veritate_model.ffn
            ckpt_args["heads"]  = veritate_model.heads
            ckpt_args["seq"]    = veritate_model.seq
            ckpt_args.setdefault("description", args.description)
            ckpt_path = save.save(veritate_model, name, step, optimizer=opt, args=ckpt_args)
            print("checkpoint + hooks: " + ckpt_path, flush=True)

    print("done.", flush=True)
