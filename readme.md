# Trainers

Trainers are how you train, refine, tune, and distill AI models in Veritate. Each one is a small folder with a script that does the work and a `manifest.json` that tells the dashboard about it. Drop a trainer in here and it shows up in the Training tab next time you open it.

The full signature-level contract is in [`documentation/trainers/contract.md`](../documentation/trainers/contract.md). This file is the friendly tour.

## How a trainer is put together

A trainer lives in its own folder under `trainers/<name>/`. The two files that matter are:

- **`trainer.py`**. The script that runs when you click *train*.
- **`manifest.json`**. A short file with the trainer's name, what it does, and the default values for its settings. The dashboard reads this to build the form.

Anything else in the folder is yours to use: model code, helpers, configs, whatever. Two folder names have special meaning: `corpus/` is for training data, and `common/` (at the `trainers/` level) is where helpers shared by more than one trainer live.

## What goes in the manifest

The manifest carries trainer identity, the shape table for the size dropdown, and preset form values.

```json
{
  "name": "My Trainer",
  "description": "What this trainer does in one sentence.",
  "kind": "trainer",
  "flow": ["scratch", "continue"],
  "sizes": {
    "200m": { "layers": 16, "hidden": 1024, "ffn": 4096, "heads": 16, "params": 202000000 }
  },
  "defaults": {
    "size": "200m",
    "precision": "bf16",
    "version": "v1",
    "batch_size": 8
  }
}
```

`kind` is `"trainer"` (the only kind today).

`flow` is how the trainer starts. `"scratch"` builds a new model from random init or from a named base; `"continue"` resumes an existing model. A trainer may support either or both.

`sizes` is the shape table with exactly one entry: each trainer is standalone at one size, and the dashboard fixes the size from the trainer you pick (there is no size dropdown for plugin trainers). The key is the size label (`"80m"`, `"200b"`); the value carries `layers`, `hidden`, `ffn`, `heads`, `params`, optionally `active_params` for MoE. Trainers are named by their size (`"Veritate 200B"`).

The keys under `defaults` line up with fields the dashboard already knows how to render (`TRAINER_SCHEMA` in `veritate_mri/web/index.js`). Pick the ones you have an opinion about and skip the rest. The full list of recognized keys is in the canonical contract.

### Reserved trainer flags

A handful of `defaults` keys are *reserved*. The dashboard recognizes them by name and renders them with special affordances. Use the reserved name when your trainer supports the behavior; do not invent a near-synonym.

| key | type | what it means |
|---|---|---|
| `qat_enabled` | bool | wrap matmul weights, embeddings, RMSNorm, and the residual add with fake-quant ops (per-tensor maxabs INT8 weights, scale-32 INT8 activations, scale-64 INT8 LN weights) using a straight-through estimator. Result: a checkpoint whose INT8 export lands with `act_boost=1` and runs cleanly on the engine. |
| `quant_mode` | string | weight quant scheme: `"int8"` (default), `"int4"` (2x density), `"ternary"` (BitNet b1.58, ~5x density). Activation and RMSNorm quant stay INT8 regardless. |
| `use_8bit_adam` | bool | construct the optimizer as `bitsandbytes.optim.AdamW8bit`. Lets ~1B-class training fit on 12 GB-class consumer GPUs. |

If you need a new reserved flag, update the canonical contract and the dashboard schema in the same commit.

## Hooking into the dump system (required for trainers)

This is the contract that makes your training run show up in the dashboard. The Training tab, the loss chart, the brain panels, the lens, the candidates, the concept atlas. They all read from the same fixed-layout files on disk. If your trainer does not write those files, the dashboard sees nothing for your model.

**The whole surface you need lives in one module.**

```python
from veritate_core.plugin import save, paths, model, qat, get_teacher_client
```

`veritate_core.plugin` is the only thing a trainer is allowed to import from outside its own folder. The two calls below are the ones every trainer makes.

### Per step: `save.append_train_row`

Call this at every logging step. It appends one row to `models/<name>/train.csv`, which is what the loss chart reads.

```python
save.append_train_row(
    name,                       # model dir name
    step,                       # int
    "train",                    # or "val"
    float(loss.item()),         # required
    lr=lr,                      # optional
    grad_norm=float(gn),        # optional
    tok_per_s=tok_per_s,        # optional
    wall_s=time.time() - t0,    # optional
    seed=args.seed,             # optional
)
```

Cheap. Call it for both train and val rows. The loss curve, throughput chart, learning-rate chart, and gradient-norm chart all read from this one CSV.

### Per checkpoint: `save.save`

Call this every time you want to save a checkpoint. It writes the `.pt` file AND runs the full dump suite. One call, thirteen artifacts on disk.

```python
ckpt_path = save.save(
    model,                      # torch.nn.Module with a vanilla state_dict()
    name,                       # model dir name
    step,                       # int
    optimizer=opt,              # optional, embedded in the .pt
    args=vars(args),            # dict; description is auto-derived if not set
)
```

What lands on disk after one `save.save()` call:

```
models/<name>/
  checkpoints/
    step_<N>.pt                       the torch checkpoint
  hooks/
    step_<N>/
      probe.json                      top-K firing neurons per layer on the canonical prompt
      lens.npz                        logit-lens projections per layer
      classroom.json                  param count, INT8/INT4 byte budget, weight-delta L2, alive neurons
      grades.json                     reading-grade rubric scores
      math.json                       arithmetic-eval rubric scores
      grammar.json                    grammar-eval rubric scores
      reasoning.json                  reasoning-eval rubric scores
      concepts.json                   top concept neurons per layer
      surprise.json                   per-byte surprise on the canonical prompt
      quant_kl.json                   KL between fp32 logits and a quantised projection
      writing_health.json             repetition + vocab-spread telemetry
      reading_comprehension.json      multi-prompt comprehension rubric scores
      generation.json                 full per-byte frame stream for the canonical prompt
```

Field schemas for every artifact live in [`documentation/hooks/contract.md`](../documentation/hooks/contract.md). That file is the contract; rename a field there and in the dumper in the same commit.

If a particular dump fails (out of memory, missing corpus, bad shape) it gets logged and skipped — the checkpoint still lands. Pass `dump_set={"surprise", "quant_kl"}` to skip specific dumps deliberately.

### What `save.save` requires from your model

- `model` is a `torch.nn.Module` whose `state_dict()` returns vanilla, unwrapped weights for a Veritate base. The dump suite assumes vanilla Veritate shapes when it builds the probe and lens passes.
- If you train a wrapper around a Veritate base (an adapter, a holographic head, a side network), call `save.save(model.base, ...)` so the dumps see the standard model. Save the wrapper state to a sidecar `.pt` next to the standard checkpoint.
- `args` should include a `description`. If it doesn't, `save.save` auto-derives one from `args` (corpus, size, precision, version, shape, training mode, seed). If nothing usable exists in `args` and `config.json` doesn't already have one, it raises.

### Other helpers in `save`

| call | what it does |
|---|---|
| `save.compose_name(user_name, size)` | build the canonical model name `<slug>_<size>` (e.g. `chatty_otter_85m`). Legacy 4-arg form `compose_name(corpus, size, precision, version)` still works. |
| `save.hash_corpus(stem)` | sha256 of the corpus train (and val if present) `.bin` files; record in `config.json` to fingerprint the data |
| `save.require_description(desc)` | trims and validates a description string; raises if empty |
| `save.resolve_corpus(stem)` | returns `(train_path, val_path)` for a corpus stem; searches shared then bundled |
| `save.truncate_train_csv_at(name, resume_step)` | drop train.csv rows past `resume_step` after loading a checkpoint, so a resumed run doesn't double-log step numbers |

### The `paths` namespace

`paths` is a pure read-only helper. It builds on-disk paths so trainers do not assemble them by hand.

| call | returns |
|---|---|
| `paths.model_dir(name)` | absolute path to `models/<name>/` |
| `paths.config_path(name)` | `models/<name>/config.json` |
| `paths.train_csv_path(name)` | `models/<name>/train.csv` |
| `paths.checkpoints_dir(name)` | `models/<name>/checkpoints/` |
| `paths.checkpoint_path(name, step)` | `models/<name>/checkpoints/step_<N>.pt` |
| `paths.hooks_dir(name)` | `models/<name>/hooks/` |
| `paths.hook_step_dir(name, step)` | `models/<name>/hooks/step_<N>/` |
| `paths.hook_artifact_path(name, step, artifact)` | path to one of the thirteen dump files |
| `paths.corpus_dir()` | `trainers/corpus/` |
| `paths.corpus_train_path(stem)` | `trainers/corpus/<stem>_train.bin` |
| `paths.corpus_val_path(stem)` | `trainers/corpus/<stem>_val.bin` |

### What trainers must not do

- Do not import from `veritate_mri.*` directly. Use `veritate_core.plugin.save` and `veritate_core.plugin.paths`.
- Do not write outside `models/<name>/`. The dashboard reads from a fixed layout; writing elsewhere is invisible to it.
- Do not edit `config.json` after `save.save` has bootstrapped it, except via fields the contract defines.
- Do not invent your own dump artifacts. The dashboard only renders the thirteen in `HOOK_ARTIFACTS`. If you need a new field, add it through the [hooks contract](../documentation/hooks/contract.md) update process.

## Making a new trainer

Easiest way: copy one of the shipped trainers under `trainers/` whose flow matches what you want, rename the folder, and edit.

1. Edit the manifest. Set the name and description, fill in the `sizes` table, fill in any `defaults` you care about.
2. Write `trainer.py`. Take the CLI args the dashboard passes in, run your training loop, and at every log step call `save.append_train_row(...)`, at every checkpoint call `save.save(...)`.
3. Open the dashboard and refresh the trainers list. Your trainer should appear in the trainer dropdown.

If something is missing (the trainer does not show up, the form looks empty, training will not start) most of the time it is that `manifest.json` is in the wrong place or has a typo. The dashboard logs will tell you.

## Training data

Your trainer reads training data from `.bin` files. Two places to put them:

- **Shared:** `trainers/corpus/<name>_train.bin` (and optionally `<name>_val.bin`). Anything here is visible to every trainer.
- **Bundled:** `trainers/<your_trainer>/corpus/<name>_train.bin`. Stays with the trainer, only that trainer sees it.

Use `save.resolve_corpus(stem)` to locate the files. It returns `(train_path, val_path)` and searches the shared folder first, then the calling trainer's bundled folder. Raises `FileNotFoundError` if no train file exists; `val_path` is `None` when there is no val file.

Build scripts that produce these files live with their consumer. A builder used by one trainer lives in that trainer's folder; a builder shared by many lives in `trainers/common/`. Output `.bin` files always land in `trainers/corpus/` (shared) or `trainers/<trainer>/corpus/` (bundled).

## Updating

Hit *Sync* in the dashboard's Settings tab to pull the latest trainers.
