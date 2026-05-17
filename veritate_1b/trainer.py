# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_mega trainer plugin. Trains the MEGA MoE byte-level Veritate
#   (top-1 router, N independent expert FFNs per block) end-to-end on a flat
#   byte corpus.
# - MEGA blocks are stateless like vanilla Veritate. There is no recurrent
#   state across timesteps, so the chunked_step pattern from veritate_200m
#   buys nothing here. We do one forward per step on (batch, seq) inputs,
#   which is faster and keeps the gradient path clean.
# - Perf choices vs veritate_200m: foreach AdamW on non-CUDA, no per-step
#   .item() / scalar bool syncs in the inner loop, prefetch the next batch
#   on a worker thread so data load overlaps the GPU step.
# - Boolean defaults from manifest.json are honored explicitly (the platform-
#   wide argparse store_true behavior would otherwise ignore them).
# - QAT support: same machinery as veritate_200m. Two-phase recipe is the
#   recommended path; phase A trains non-QAT to coherence, phase B resumes
#   with --qat_enabled to fine-tune the fake-quanted matmuls / embeddings /
#   norms / residual adds for clean v11 export.
# plugins/veritate_mega/plugin.py
# ------------------------------------------------------------------------------------

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from veritate_core.plugin import save, paths, qat as qat_helpers

import mega as _mega


with open(os.path.join(HERE, "manifest.json"), "r", encoding="utf-8") as _f:
    MANIFEST = json.load(_f)


# ------------------------------------------------------------------------------------
# Constants

# Shape table comes from manifest.json's `sizes` block. Active params per
# token (top-1 routing) is the attention path plus one expert FFN per block;
# total params (`params`) and per-token active params (`active_params`) live
# in the manifest alongside the shape so the dashboard's MoE-aware VRAM
# estimator reads from one place.
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
# Functions

def parse_args():
    ap = argparse.ArgumentParser(description=MANIFEST.get("description", ""))
    ap.add_argument("--name",        type=str, default="")
    ap.add_argument("--corpus",      type=str, default="")
    ap.add_argument("--description", type=str, default="")
    ap.add_argument("--resume",      type=str, default="")
    for k, v in MANIFEST.get("defaults", {}).items():
        if k in ("name", "corpus", "description", "resume"):
            continue
        if isinstance(v, bool):
            # Dashboard form passes booleans as bare flags (no value).
            # BooleanOptionalAction handles `--flag` and `--no-flag` and
            # respects the manifest default when the flag is not passed.
            ap.add_argument("--" + k, action=argparse.BooleanOptionalAction, default=bool(v))
        elif isinstance(v, int):
            ap.add_argument("--" + k, type=int,   default=v)
        elif isinstance(v, float):
            ap.add_argument("--" + k, type=float, default=v)
        else:
            ap.add_argument("--" + k, type=str,   default=str(v))
    # Dashboard renders the full TRAINER_SCHEMA for every plugin; flags this
    # trainer does not implement arrive on argv. Drop them silently.
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
    """LR schedule. WSD (Warmup-Stable-Decay) keeps base_lr flat after
    warmup until the last `wsd_decay_frac` of training, then decays to
    min_lr under `wsd_decay_kind` ∈ {sqrt, linear, cosine}. See 800m
    plugin's lr_at for the same wiring."""
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


class _Prefetcher:
    """Single-thread prefetcher that overlaps the next batch's numpy fancy-
    indexing with the current step's GPU work. The main loop calls .next() to
    get the current batch and immediately receives it; meanwhile the worker is
    already preparing the one after. CPU work is small (a few MB of memcopy)
    but on MPS the main thread otherwise stalls the GPU between steps."""

    def __init__(self, draw_fn):
        self._draw = draw_fn
        self._lock = threading.Lock()
        self._cv   = threading.Condition(self._lock)
        self._next = None
        self._die  = False
        self._thr  = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while True:
            with self._cv:
                while (self._next is not None) and not self._die:
                    self._cv.wait()
                if self._die:
                    return
            batch = self._draw()
            with self._cv:
                self._next = batch
                self._cv.notify_all()

    def next(self):
        with self._cv:
            while self._next is None:
                self._cv.wait()
            batch = self._next
            self._next = None
            self._cv.notify_all()
        return batch

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
        return torch.from_numpy(toks), torch.from_numpy(tgts)

    return draw, N


def pick_device():
    forced = (os.environ.get("VERITATE_DEVICE") or "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        ok = (forced == "cpu") \
             or (forced == "cuda" and torch.cuda.is_available()) \
             or (forced == "mps"  and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if ok: return forced
        print(f"[veritate_1b] requested device={forced!r} unavailable; using cpu", flush=True)
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_config(name, args, base_cfg, n_params, n_active_est, corpus_hash):
    cfg_path = paths.config_path(name)
    os.makedirs(paths.model_dir(name), exist_ok=True)
    ta = vars(args).copy()
    if corpus_hash:
        ta["corpus_sha256"] = corpus_hash.get("train_sha256")
        ta["corpus_bytes"]  = corpus_hash.get("train_bytes")
    shape = dict(base_cfg)
    shape["seq"]         = args.seq
    shape["vocab"]       = _mega.VOCAB_BYTE_LEVEL
    shape["n_experts"]   = args.n_experts
    shape["router_topk"] = args.router_topk
    qat_on = bool(getattr(args, "qat_enabled", False))
    cfg = {
        "name": name,
        "description": args.description,
        "kind": "trainer",
        "plugin": "veritate_mega",
        "vocab": _mega.VOCAB_BYTE_LEVEL,
        "shape": shape,
        "training":  ("qat" if qat_on else ""),
        "qat_source": (args.resume if (qat_on and args.resume) else ""),
        "training_args": ta,
        "n_params_total":  n_params,
        "n_params_active": n_active_est,
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
    # Resume into a different shape (e.g. n_experts changed) is the most
    # likely cause of either set being non-trivially populated. Refuse rather
    # than silently re-randomizing modules.
    SHAPE_HINT = ("router", "experts_up", "experts_down")
    shape_missing = [k for k in missing if any(s in k for s in SHAPE_HINT)]
    shape_extra   = [k for k in extra   if any(s in k for s in SHAPE_HINT)]
    if shape_missing or shape_extra:
        raise RuntimeError(
            "checkpoint shape mismatch on resume; will not silently re-init MoE weights.\n"
            "  missing in checkpoint: " + str(sorted(shape_missing)[:6]) + "\n"
            "  extra in checkpoint:   " + str(sorted(shape_extra)[:6])
        )
    model.load_state_dict(sd, strict=False)
    return ckpt.get("optimizer")


@torch.no_grad()
def evaluate(model, val_draw, n_iters, amp_dtype, device, device_type):
    model.eval()
    losses = []
    for _ in range(n_iters):
        toks, tgts = val_draw()
        toks = toks.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            _, loss, _ = model(toks, targets=tgts)
        if loss is not None:
            losses.append(loss.detach())
    model.train()
    if not losses:
        return None
    return float(torch.stack(losses).mean().item())


def estimate_active_params(shape, n_experts, vocab):
    """Approx active params per token under top-1 routing: embedding + every
    block's attention + (1 expert FFN) + RMSNorms + LM head (tied)."""
    h, L, ffn, heads = shape["hidden"], shape["layers"], shape["ffn"], shape["heads"]
    per_block_attn   = 4 * h * h  # qkv (3h*h) + proj (h*h)
    per_block_ffn1   = 2 * h * ffn  # one expert: up + down
    per_block_router = h * n_experts
    per_block_norm   = 2 * h
    per_block        = per_block_attn + per_block_ffn1 + per_block_router + per_block_norm
    embed   = vocab * h          # tied with lm_head, count once
    pos     = shape.get("seq", 1024) * h
    n_out   = h
    return int(L * per_block + embed + pos + n_out)


def main():
    args = parse_args()
    resume_mode = bool(args.resume)
    qat_enabled = bool(getattr(args, "qat_enabled", False))
    qat_source  = args.resume if (resume_mode and qat_enabled) else None
    if resume_mode:
        apply_resume_overrides(args, sys.argv)
        if qat_enabled:
            args.qat_enabled = True
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
    if int(args.router_topk) != 1:
        raise ValueError("only router_topk=1 is supported in this build")

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

    train_path, val_path = save.resolve_corpus(args.corpus)
    print("corpus train: " + train_path, flush=True)
    if val_path:
        print("corpus val:   " + val_path, flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Apple Silicon / unified-memory specific tuning. Defaults are conservative
    # for laptops; on an Ultra-class box we want the memory cap off and the
    # CPU thread count low so the prefetcher doesn't fight the main thread.
    if not os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(min(8, max(2, (os.cpu_count() or 4) // 4)))
    device = pick_device()
    print(f"device: {device}", flush=True)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None

    shape = SIZE_PRESETS[args.size]
    model = _mega.Mega(
        vocab=_mega.VOCAB_BYTE_LEVEL,
        hidden=shape["hidden"], layers=shape["layers"],
        ffn=shape["ffn"], heads=shape["heads"], seq=args.seq,
        n_experts=args.n_experts, router_topk=args.router_topk,
    )
    if qat_enabled:
        qat_helpers.set_qat(model, True)
        print("QAT: enabled (fake-quant matmuls + embeddings + RMSNorm + residual adds)", flush=True)

    if args.use_act_ckpt:
        print("activation checkpointing: ENABLED", flush=True)
        for blk in model.blocks:
            blk.forward = (lambda fwd: lambda x: torch.utils.checkpoint.checkpoint(fwd, x, use_reentrant=False))(blk.forward)

    model.to(device)
    n_params       = sum(p.numel() for p in model.parameters())
    n_active_est   = estimate_active_params(shape, args.n_experts, _mega.VOCAB_BYTE_LEVEL)
    print("device: " + device + "  precision: " + args.precision, flush=True)
    print("params total:  " + str(n_params), flush=True)
    print("params active: ~" + str(n_active_est) + " (top-1)", flush=True)
    print("shape:  hidden=" + str(shape["hidden"]) + " layers=" + str(shape["layers"])
          + " ffn=" + str(shape["ffn"]) + " heads=" + str(shape["heads"])
          + " seq=" + str(args.seq) + " n_experts=" + str(args.n_experts), flush=True)

    resume_step = 0
    resume_opt_state = None
    if qat_source is not None:
        src_step = latest_checkpoint_step(qat_source)
        print("QAT load: " + qat_source + " step " + str(src_step) + "  -> new model " + name, flush=True)
        load_resume_state(model, qat_source, src_step, device)
        write_config(name, args, shape, n_params, n_active_est, corpus_hash=None)
        print("wrote: " + paths.config_path(name), flush=True)
    elif resume_mode:
        resume_step = latest_checkpoint_step(name)
        print("resume: " + name + "  from step " + str(resume_step), flush=True)
        resume_opt_state = load_resume_state(model, name, resume_step, device)
    else:
        print("hashing corpus (one-time)...", flush=True)
        corpus_hash = save.hash_corpus(args.corpus)
        print("corpus sha256: " + corpus_hash.get("train_sha256", "?")[:16] + "...  bytes=" + str(corpus_hash.get("train_bytes")), flush=True)
        write_config(name, args, shape, n_params, n_active_est, corpus_hash)
        print("wrote: " + paths.config_path(name), flush=True)

    train_draw, train_n = make_data_loader(train_path, args.batch_size, args.seq, args.seed)
    val_draw = None
    if val_path:
        val_draw, _ = make_data_loader(val_path, args.batch_size, args.seq, args.seed + 1)
    print("train corpus bytes: " + str(train_n) + "  per-step tokens: " + str(args.batch_size * args.seq), flush=True)

    train_pf = _Prefetcher(train_draw)

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

    aux_coef = float(args.router_aux_loss_coef)

    t0 = time.time()
    last_log = t0
    last_log_step = resume_step
    start_step = resume_step + 1
    try:
        for step in range(start_step, args.total_steps + 1):
            lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                       wsd_decay_frac=getattr(args, "wsd_decay_frac", 0.1),
                       wsd_decay_kind=getattr(args, "wsd_decay_kind", "sqrt"),
                       schedule=args.lr_schedule)
            for g in opt.param_groups:
                g["lr"] = lr

            toks, tgts = train_pf.next()
            toks = toks.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)

            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _, loss, aux = model(toks, targets=tgts)
            total_loss = loss if aux is None else (loss + aux_coef * aux)
            total_loss.backward()
            gn_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step % args.log_every == 0 or step == 1:
                # one sync, here, for logging only.
                loss_v = float(loss.detach().item())
                aux_v  = (float(aux.detach().item()) if aux is not None else 0.0)
                gn_v   = float(gn_t.detach().item())
                now = time.time()
                elapsed = now - t0
                window_s = max(1e-6, now - last_log)
                window_steps = step - last_log_step
                tok_per_s = window_steps * args.batch_size * args.seq / window_s
                print("step " + str(step) + "  loss " + format(loss_v, ".4f")
                      + "  aux " + format(aux_v, ".4f")
                      + "  lr " + format(lr, ".2e")
                      + "  gn " + format(gn_v, ".3f")
                      + "  tok/s " + format(tok_per_s, ".0f")
                      + "  elapsed " + format(elapsed, ".0f") + "s", flush=True)
                save.append_train_row(name, step, "train", loss_v,
                                      lr=lr, grad_norm=gn_v,
                                      tok_per_s=tok_per_s, wall_s=elapsed, seed=args.seed)
                last_log = now
                last_log_step = step

            if val_draw is not None and step % args.eval_every == 0:
                v = evaluate(model, val_draw, args.eval_iters, amp_dtype, device, device)
                if v is not None:
                    print("step " + str(step) + "  val_loss " + format(v, ".4f"), flush=True)
                    save.append_train_row(name, step, "val", v, lr=lr,
                                          wall_s=time.time() - t0, seed=args.seed)

            if step % args.ckpt_every == 0 or step == args.total_steps:
                ckpt_args = vars(args).copy()
                ckpt_args["vocab"]       = model.vocab
                ckpt_args["hidden"]      = model.hidden
                ckpt_args["layers"]      = model.layers
                ckpt_args["ffn"]         = model.ffn
                ckpt_args["heads"]       = model.heads
                ckpt_args["seq"]         = model.seq
                ckpt_args["n_experts"]   = model.n_experts
                ckpt_args["router_topk"] = model.router_topk
                ckpt_args.setdefault("description", args.description)
                ckpt_path = save.save(model, name, step, optimizer=opt, args=ckpt_args)
                print("checkpoint + hooks: " + ckpt_path, flush=True)
    finally:
        train_pf.close()

    print("done.", flush=True)


if __name__ == "__main__":
    main()
