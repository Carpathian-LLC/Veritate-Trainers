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
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

from veritate_core.plugin import save, paths, model as _model_mod, qat as qat_helpers, multicorpus, bench, hardware
# Optional: shared optimizer builder (muon) and patched-trunk variant.
# Feature-detect so this trainer still runs on an older platform.
try:
    from veritate_core.plugin import optim as optim_helpers
except ImportError:
    optim_helpers = None
try:
    from veritate_core.plugin import model_patched as _model_patched_mod
    VeritatePatched = _model_patched_mod.VeritatePatched
except ImportError:
    VeritatePatched = None
try:
    from veritate_core.plugin import model_recurrent as _model_recurrent_mod
    VeritateRecurrent = _model_recurrent_mod.VeritateRecurrent
except ImportError:
    VeritateRecurrent = None
try:
    from veritate_core.plugin import model_memory as _model_memory_mod
    VeritateMemory = _model_memory_mod.VeritateMemory
except ImportError:
    VeritateMemory = None
try:
    from veritate_core.plugin import slm as slm_helpers
except ImportError:
    slm_helpers = None
# Optional: NVMe optimizer paging needs a platform new enough to expose the memory
# planner/executor offload APIs. Feature-detect so this trainer still runs on an
# older platform (it just falls back to the pre-paging behavior).
try:
    from veritate_core.plugin import mem_planner, mem_executor
    _MEM_PAGING = (hasattr(mem_executor, "make_optimizer")
                   and hasattr(mem_executor, "OFFLOAD_TIERS")
                   and hasattr(mem_planner, "plan_training_memory")
                   and hasattr(bench, "plan_result"))
except Exception:
    mem_planner = None
    mem_executor = None
    _MEM_PAGING = False

Veritate         = _model_mod.Veritate
VOCAB_BYTE_LEVEL = _model_mod.VOCAB_BYTE_LEVEL

append_train_row       = save.append_train_row
compose_name           = save.compose_name
hash_corpus            = save.hash_corpus
require_description    = save.require_description
resolve_corpus         = save.resolve_corpus
truncate_train_csv_at  = save.truncate_train_csv_at


# ------------------------------------------------------------------------------------
# Constants

SHAPE_FIELDS    = ("layers", "hidden", "ffn", "heads")
BASE_CKPT_PREFIX = "step_"
BASE_CKPT_SUFFIX = ".pt"

LR_SCHEDULES    = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS = ("sqrt", "linear", "cosine")
PRECISIONS      = ("fp32", "bf16")

STATE_CARRIES      = ("off", "chunks")
STATE_CARRY_CHUNKS = "chunks"
STATE_CARRY_TRUNKS = ("hybrid", "hybrid_moe", "recurrent")

# Weights, grads, and Adam moments are all fp32 here (autocast keeps fp32 master
# weights even at bf16 precision), so the memory planner sizes its buckets in fp32.
PLAN_DTYPE      = "fp32"
PAGED_STATE_DIR = "optim_state"


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
    "optimizer": "adamw",
    "trunk": "dense",
    "slm_ref": "",
    "state_rule": "gla",
    "state_carry": "off",
    "loss_mask": "off",
}
RESERVED_FLOAT_FLAGS = {
    "l1_lambda": 0.0,
    "slm_keep": 0.6,
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


# A corpus this dense in ChatML turn markers is a chat/SFT set, not free prose.
# Training one WITHOUT loss_mask=assistant computes loss over the user's turns too,
# so the model learns to continue the user rather than answer them. That silently
# produces a fluent model that ignores the question, so the trainer refuses to
# start until the caller decides explicitly.
TRAINING_KIND_CHAT      = "chat"   # stamped on config.json so the serve path knows
                                   # this model expects ChatML framing, instead of
                                   # guessing and mismatching what it was trained on
CHATML_DENSITY_SFT      = 0.20     # marker-bearing sample windows / windows read
CHATML_PROBE_WINDOWS    = 64
CHATML_PROBE_WINDOW_LEN = 8192

CHATML_ASSISTANT_OPEN = b"<|im_start|>assistant\n"
CHATML_IM_START       = b"<|im_start|>"
CHATML_IM_END         = b"<|im_end|>"


def _keep_assistant(row_bytes):
    n = len(row_bytes)
    keep = np.zeros(n, dtype=bool)
    ao, ist, ie = CHATML_ASSISTANT_OPEN, CHATML_IM_START, CHATML_IM_END
    la, li, le = len(ao), len(ist), len(ie)
    i = 0
    in_asst = False
    while i < n:
        if row_bytes[i:i + la] == ao:
            in_asst = True; i += la; continue
        if row_bytes[i:i + li] == ist:
            in_asst = False; i += li; continue
        if in_asst and row_bytes[i:i + le] == ie:
            keep[i:i + le] = True; i += le; in_asst = False; continue
        keep[i] = in_asst
        i += 1
    return keep


def apply_role_mask(toks, tgts):
    tk = toks.cpu().numpy().astype(np.uint8)
    tg = tgts.clone()
    B, N = tk.shape
    for b in range(B):
        rb = tk[b].tobytes()
        if CHATML_IM_START not in rb:
            continue
        keep = _keep_assistant(rb)
        active = np.zeros(N, dtype=bool)
        active[:-1] = keep[1:]
        tg[b][~torch.from_numpy(active)] = -1
    return tg


def chunked_step(model, tokens, targets, seq, amp_dtype, *, backward=False, bptt_window=1,
                 device_type="cuda", l1_lambda=0.0, slm_ref=None, slm_keep=0.6,
                 state_carry="off"):
    B, total_len = tokens.shape
    n_chunks = max(1, total_len // seq)
    K = max(1, int(bptt_window))
    loss_sum = 0.0
    n_valid  = 0
    window_losses = []
    capture_l1 = bool(getattr(model, "capture_l1", False)) and l1_lambda > 0.0
    # Memory trunks carry fast-weight state across the contiguous chunks of one
    # step (reset per step): the loss then rewards reading memory beyond the
    # attention window. Non-memory models lack forward_carry and take the
    # plain path.
    mem_carry = hasattr(model, "forward_carry")
    if mem_carry:
        model.reset_memory()
    # state_carry=chunks: recurrent global state threads left-to-right through the
    # step's contiguous chunks via forward_streaming (positions stay window-local).
    # Grads flow through the carry inside a bptt window; carry detached at each
    # backward. State resets per step (rows are unrelated stream offsets).
    rec_carry  = state_carry == STATE_CARRY_CHUNKS
    rec_states = None
    for cstart in range(0, total_len, seq):
        cend = min(cstart + seq, total_len)
        ct = tokens[:, cstart:cend]
        cg = targets[:, cstart:cend]
        if ct.size(1) < 2:
            break
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            if rec_carry:
                logits, rec_states = model.forward_streaming(ct, rec_states)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), cg.reshape(-1), ignore_index=-1)
            else:
                logits, loss = model.forward_carry(ct, targets=cg) if mem_carry else model(ct, targets=cg)
            if slm_ref is not None:
                loss = slm_helpers.selective_loss(slm_ref, ct, cg, logits, keep_frac=slm_keep)
        if loss is None or not torch.isfinite(loss):
            continue
        if capture_l1:
            l1 = model.post_l1_sum()
            if l1 is not None:
                loss = loss + l1_lambda * l1
        if backward:
            aux = model.moe_aux_sum() if hasattr(model, "moe_aux_sum") else None
            if aux is not None:
                loss = loss + aux
        loss_sum += float(loss.detach().item())
        n_valid  += 1
        if backward:
            window_losses.append(loss)
            window_full = len(window_losses) >= K
            last_chunk  = (cstart + seq) >= total_len
            if window_full or last_chunk:
                (torch.stack(window_losses).sum() / n_chunks).backward()
                window_losses = []
                if mem_carry:
                    model.carry_memory()
                if rec_states is not None:
                    rec_states = [{k: v.detach() for k, v in st.items()} for st in rec_states]
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
        "training":  ("qat" if qat_on else (args.training_kind or "")),
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
def evaluate(model, val_draw, n_iters, seq, amp_dtype, bptt_window, device_type="cuda",
             state_carry="off"):
    model.eval()
    losses = []
    for _ in range(n_iters):
        toks, tgts = val_draw()
        toks = toks.to(next(model.parameters()).device, non_blocking=True)
        tgts = tgts.to(next(model.parameters()).device, non_blocking=True)
        loss = chunked_step(model, toks, tgts, seq, amp_dtype,
                            bptt_window=bptt_window, device_type=device_type,
                            state_carry=state_carry)
        if loss is not None:
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else None


def build_optimizer(params, args, device, plan=None, state_dir=None, model=None):
    want_muon = getattr(args, "optimizer", "adamw") == "muon"
    if want_muon and (optim_helpers is None or model is None):
        print("optimizer=muon requested but unavailable on this platform; using AdamW", flush=True)
        want_muon = False
    if want_muon:
        print("optimizer: Muon (2D hidden weights) + AdamW (emb/norms/1D)", flush=True)
        return optim_helpers.build_muon(model, args)
    if _MEM_PAGING and plan is not None and plan.tier in mem_executor.OFFLOAD_TIERS:
        print("optimizer state paged to NVMe (" + plan.tier + "); resident optimizer "
              "memory ~0, step time is disk-bound", flush=True)
        return mem_executor.make_optimizer(
            params, plan,
            lr=args.base_lr, betas=(args.beta1, args.beta2), eps=1e-6,
            weight_decay=args.weight_decay, state_dir=state_dir)
    use_8bit = bool(getattr(args, "use_8bit_adam", False))
    if use_8bit and importlib.util.find_spec("bitsandbytes") is None:
        print("8-bit AdamW requested but bitsandbytes is not installed; "
              "falling back to torch AdamW (fp32 moments, ~2x optimizer memory)", flush=True)
        use_8bit = False
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


def chatml_density(train_path):
    """Fraction of sampled windows that contain a ChatML turn marker. Sampled
    rather than scanned: a pretrain corpus can be gigabytes."""
    try:
        size = os.path.getsize(train_path)
    except OSError:
        return 0.0
    if size <= 0:
        return 0.0
    span = min(CHATML_PROBE_WINDOW_LEN, size)
    hits = 0
    with open(train_path, "rb") as f:
        for i in range(CHATML_PROBE_WINDOWS):
            off = (size - span) * i // max(1, CHATML_PROBE_WINDOWS - 1)
            f.seek(off)
            if CHATML_IM_START in f.read(span):
                hits += 1
    return hits / CHATML_PROBE_WINDOWS


def require_loss_mask_decision(args, corpus_mix, argv):
    """Refuse to train a ChatML-dense corpus until loss_mask is set explicitly.

    Forgetting it is silent: the run looks healthy, the loss falls, and the model
    comes out fluent but unable to answer a question. A warning is too easy to
    miss in a long log, so this is an error the caller has to answer."""
    if any(a == "--loss_mask" or a.startswith("--loss_mask=") for a in argv):
        return
    if getattr(args, "loss_mask", "off") != "off":
        return
    for _stem, _val, train_path, _w in _iter_mix_paths(corpus_mix):
        density = chatml_density(train_path)
        if density >= CHATML_DENSITY_SFT:
            raise SystemExit(
                "refusing to start: corpus '" + str(args.corpus) + "' is "
                + str(int(density * 100)) + "% ChatML turn markers, which makes it a "
                "chat/SFT corpus, but --loss_mask was never set.\n"
                "  Without it, loss is computed over the USER's turns too and the model "
                "learns to continue the user instead of answering.\n"
                "  Pass --loss_mask assistant   (what you almost certainly want for SFT)\n"
                "  or   --loss_mask off         (to train on every byte on purpose)")


def _iter_mix_paths(corpus_mix):
    """Yield (stem, val_path, train_path, weight) for a resolved multicorpus mix,
    tolerating the tuple shapes the resolver has used across versions."""
    for entry in corpus_mix or []:
        if not isinstance(entry, (tuple, list)):
            continue
        paths_in = [x for x in entry if isinstance(x, str) and x.endswith(".bin")]
        train_path = None
        for cand in paths_in:
            if cand.endswith("_train.bin"):
                train_path = cand
                break
        if train_path is None and paths_in:
            train_path = paths_in[-1]
        if train_path:
            yield (None, None, train_path, None)


def detect_training_kind(corpus_mix):
    """What framing did this model learn? Stamped on config.json so serving reads
    it instead of assuming. A model trained on plain prose and then served ChatML
    produces confident nonsense, which is exactly how chat_200m broke."""
    for _stem, _val, train_path, _w in _iter_mix_paths(corpus_mix):
        if chatml_density(train_path) >= CHATML_DENSITY_SFT:
            return TRAINING_KIND_CHAT
    return ""


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
    args.training_kind = ""
    if not args.bench:
        _corpus_mix = multicorpus.resolve_and_weight(args.corpus, resolve_corpus)
        val_path    = _corpus_mix[0][1]
        print("corpus mix:   " + multicorpus.format_mix_summary(_corpus_mix), flush=True)
        if val_path:
            print("corpus val:   " + val_path, flush=True)
        require_loss_mask_decision(args, _corpus_mix, sys.argv)
        args.training_kind = detect_training_kind(_corpus_mix)
        if args.training_kind:
            print("training kind: " + args.training_kind
                  + " (config.json records the framing this model expects)", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = hardware.pick_device()
    print("device: " + device, flush=True)
    amp_dtype = hardware.resolve_precision(args.precision, device)

    shape = size_presets[args.size]

    # Plan the memory ladder from the size preset BEFORE building the model: a model
    # whose weights+grads exceed the budget cannot be built (the allocation OOMs/
    # SIGKILLs), so refuse with the floor numbers instead of hanging. bench plans at
    # batch 1 (the ramp explores batch); a real run plans at its configured batch.
    mem_plan = None
    if _MEM_PAGING and shape.get("params"):
        plan_batch = 1 if args.bench else args.batch_size
        mem_plan = mem_planner.plan_training_memory(
            shape["params"], shape["hidden"], shape["layers"], shape["ffn"],
            plan_batch, args.seq, PLAN_DTYPE)
        print("mem plan: " + mem_planner.format_plan(mem_plan), flush=True)
        if not mem_plan.fits:
            if args.bench:
                print("BENCH_RESULT " + json.dumps(bench.plan_result(mem_plan, device, args.seq)), flush=True)
            elif mem_plan.params_bytes + mem_plan.grads_bytes > mem_plan.budget_bytes:
                print("INFEASIBLE: weights+grads alone exceed the memory budget; paging "
                      "the optimizer cannot help. Use a smaller size or a larger-memory "
                      "machine.", flush=True)
            else:
                print("INFEASIBLE at this batch/seq: activations push the run over budget "
                      "even with checkpointing. Reduce batch_size or seq.", flush=True)
            return

    activation = getattr(args, "activation", "gelu") or "gelu"
    l1_lambda  = float(getattr(args, "l1_lambda", 0.0) or 0.0)
    trunk      = getattr(args, "trunk", "dense") or "dense"
    model_cls  = Veritate
    model_kwargs = {}
    if trunk == "patched":
        if VeritatePatched is None:
            raise ValueError("trunk=patched requested but platform lacks model_patched")
        model_cls = VeritatePatched
    elif trunk == "hybrid":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent"}
    elif trunk == "hybrid_moe":
        if VeritatePatched is None:
            raise ValueError("trunk=hybrid_moe requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_mixer": "recurrent", "global_ffn": "moe"}
    elif trunk == "looped":
        if VeritatePatched is None:
            raise ValueError("trunk=looped requested but platform lacks model_patched")
        model_cls = VeritatePatched
        model_kwargs = {"global_loops": 4}
    elif trunk == "recurrent":
        if VeritateRecurrent is None:
            raise ValueError("trunk=recurrent requested but platform lacks model_recurrent")
        model_cls = VeritateRecurrent
    elif trunk == "memory":
        if VeritateMemory is None:
            raise ValueError("trunk=memory requested but platform lacks model_memory")
        model_cls = VeritateMemory
    state_rule = getattr(args, "state_rule", "gla") or "gla"
    if trunk in ("hybrid", "hybrid_moe", "recurrent") and state_rule != "gla":
        model_kwargs["state_rule"] = state_rule
    state_carry = getattr(args, "state_carry", "off") or "off"
    if state_carry not in STATE_CARRIES:
        raise ValueError("unknown state_carry: " + str(state_carry)
                         + " (valid: " + ", ".join(STATE_CARRIES) + ")")
    if state_carry == STATE_CARRY_CHUNKS and trunk not in STATE_CARRY_TRUNKS:
        raise ValueError("state_carry=chunks requires trunk in: " + ", ".join(STATE_CARRY_TRUNKS))
    veritate_model = model_cls(
        vocab=VOCAB_BYTE_LEVEL,
        hidden=shape["hidden"], layers=shape["layers"],
        ffn=shape["ffn"], heads=shape["heads"], seq=args.seq,
        activation=activation,
        capture_l1=(l1_lambda > 0.0),
        **model_kwargs,
    )
    print(f"activation: {activation}  l1_lambda: {l1_lambda}  trunk: {trunk}"
          f"  state_carry: {state_carry}", flush=True)
    if qat_enabled:
        qat_helpers.set_qat(veritate_model, True)
        print("QAT: enabled (fake-quant matmuls + embeddings + RMSNorm + residual adds)", flush=True)

    if _MEM_PAGING and mem_plan is not None and mem_plan.tier in mem_executor.CHECKPOINT_TIERS:
        mem_executor.enable_grad_checkpoint(veritate_model)
        print("activation checkpointing: ENABLED (mem plan tier " + mem_plan.tier + ")", flush=True)
    elif getattr(args, "use_act_ckpt", False):
        print("activation checkpointing: ENABLED", flush=True)
        for blk in veritate_model.blocks:
            blk.forward = (lambda fwd: lambda x: torch.utils.checkpoint.checkpoint(fwd, x, use_reentrant=False))(blk.forward)

    veritate_model.to(device)
    slm_ref = None
    _slm_name = (getattr(args, "slm_ref", "") or "").strip()
    if _slm_name:
        if slm_helpers is None:
            raise ValueError("slm_ref requested but platform lacks slm helper")
        slm_ref = slm_helpers.load_reference(paths.model_dir(_slm_name), device)
        print(f"SLM: selective loss ON, reference={_slm_name}, keep_frac={getattr(args, 'slm_keep', 0.6)}", flush=True)
    n_params = sum(p.numel() for p in veritate_model.parameters())
    print("device: " + device + "  precision: " + args.precision, flush=True)
    print("params: " + str(n_params), flush=True)
    print("shape:  hidden=" + str(shape["hidden"]) + " layers=" + str(shape["layers"])
          + " ffn=" + str(shape["ffn"]) + " heads=" + str(shape["heads"])
          + " seq=" + str(args.seq), flush=True)

    if args.bench:
        if _MEM_PAGING:
            result = bench.run(veritate_model, device, args.seq, VOCAB_BYTE_LEVEL,
                               on_progress=lambda s: print("bench: " + s, flush=True),
                               plan=mem_plan, amp_dtype=amp_dtype,
                               n_chunks=args.n_chunks, bptt_window=args.bptt_window)
        else:
            result = bench.run(veritate_model, device, args.seq, VOCAB_BYTE_LEVEL,
                               on_progress=lambda s: print("bench: " + s, flush=True),
                               amp_dtype=amp_dtype,
                               n_chunks=args.n_chunks, bptt_window=args.bptt_window)
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
        dropped = truncate_train_csv_at(name, resume_step)
        if dropped:
            print("train.csv: dropped " + str(dropped) + " stale rows past step " + str(resume_step), flush=True)
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

    if _MEM_PAGING:
        opt = build_optimizer(veritate_model.parameters(), args, device, plan=mem_plan,
                              state_dir=os.path.join(paths.model_dir(name), PAGED_STATE_DIR),
                              model=veritate_model)
    else:
        opt = build_optimizer(veritate_model.parameters(), args, device, model=veritate_model)
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
    skipped_nonfinite = 0
    for step in range(start_step, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                   schedule=args.lr_schedule,
                   wsd_decay_frac=getattr(args, "wsd_decay_frac", 0.1),
                   wsd_decay_kind=getattr(args, "wsd_decay_kind", "sqrt"))
        for g in opt.param_groups:
            g["lr"] = lr

        toks, tgts = train_draw()
        if getattr(args, "loss_mask", "off") == "assistant":
            tgts = apply_role_mask(toks, tgts)
        toks = toks.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        veritate_model.train()
        opt.zero_grad(set_to_none=True)
        loss = chunked_step(veritate_model, toks, tgts, args.seq, amp_dtype,
                            backward=True, bptt_window=args.bptt_window,
                            device_type=device, l1_lambda=l1_lambda,
                            slm_ref=slm_ref, slm_keep=float(getattr(args, "slm_keep", 0.6)),
                            state_carry=state_carry)
        if loss is None:
            skipped_nonfinite += 1
            if skipped_nonfinite % args.log_every == 0 or skipped_nonfinite == 1:
                print(f"WARNING step {step}: non-finite loss, step skipped "
                      f"({skipped_nonfinite} skipped so far)", flush=True)
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
                         args.bptt_window, device_type=device, state_carry=state_carry)
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
