# veritate_85m

Byte-level decoder at h=768 L=12 ffn=3072 heads=12 (vocab=256). ReLU FFN with
L1 sparsity penalty on post-activation, 2-byte multi-token-prediction (MTP)
head, `project_byte0` contract method for cross-model decode.

## Files

- `manifest.json` — defaults consumed by the dashboard and CLI.
- `plugin.py` — model class + trainer (self-contained, no dashboard hooks).
- `build_corpus.py` — downloads FineWeb-Edu sample-10BT from HuggingFace and
  writes `plugins/corpus/fineweb_edu_{train,val}.bin`.

## Output

`models/fineweb_edu_85m_bf16_v1_sparse/`
  - `config.json`
  - `train.csv` (one row per `--log_every` steps + one per eval)
  - `checkpoints/step_N.pt`

## Windows (PowerShell) quickstart

Assumes Python 3.13 installed via python.org, `py` launcher on PATH, NVIDIA
GPU with CUDA 12.x.

```powershell
# 1. From the repo root.
cd C:\GitHub\Veritate

# 2. Install Python deps.
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

# 3. Verify CUDA is visible.
py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Build the FineWeb-Edu .bin pair (~5 GB download + write; one-time, ~30-60 min on a fast link).
py plugins\veritate_85m\build_corpus.py

# 5. Train. Default 50000 steps, batch=24, seq=1024, bf16 on CUDA.
py plugins\veritate_85m\plugin.py --device cuda --description "first run"
```

## Monitor

The trainer prints to stdout. Tail `train.csv` for headless monitoring:

```powershell
Get-Content models\fineweb_edu_85m_bf16_v1_sparse\train.csv -Wait -Tail 5
```

## Resume

```powershell
py plugins\veritate_85m\plugin.py --device cuda --resume --description "resume"
```

## Memory notes

At default `batch=24 seq=1024 bf16`, peak VRAM is ~6-8 GB. If your GPU has
≤12 GB and you hit OOM, drop `--batch_size 16` or `--seq 512`.

## Tunables worth knowing

- `--l1_lambda` (default 1e-4): weight of the post-ReLU L1 penalty. Lower
  = less sparsity, higher quality. 0 = ablate the sparsity loss entirely.
- `--mtp_aux_weight` (default 0.5): weight of the byte-t+1 prediction head
  in the loss. 0 = trains as a pure byte-0 model with an unused MTP head.
- `--total_steps` (default 50000): ~1.2B tokens at batch=24 seq=1024.
  Chinchilla-optimal for 85M is ~2B tokens (~85k steps); the default is
  sub-Chinchilla to ship a model in 1.5-3 days. Push higher if compute
  budget allows.
