# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Build the byte-level FineWeb-Edu .bin pair from the HuggingFace
#   sample-10BT subset (the smallest curated FineWeb-Edu shard, ~3-5 GB
#   after byte-level concatenation; 10 billion BPE tokens ~= same in bytes
#   for English-dominant text).
# - Output:
#     plugins/corpus/fineweb_edu_train.bin
#     plugins/corpus/fineweb_edu_val.bin
# - Idempotent: resumable per-document. Re-running skips already-written
#   shards via a manifest at plugins/corpus/.fineweb_edu_progress.json.
# - Uses HuggingFace `datasets` library (streaming) so we never need 5+ GB
#   of intermediate parquet on disk. Network-dependent; the first run
#   downloads ~5 GB through hf-xet.
# - Document separator: a single 0x00 byte. The trainer's random-window
#   sampler will sometimes straddle a boundary; that's fine at byte level.
# plugins/veritate_85m/build_corpus.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


# ------------------------------------------------------------------------------------
# Constants

CORPUS_DIR = os.path.join(REPO_ROOT, "plugins", "corpus")
TRAIN_BIN = os.path.join(CORPUS_DIR, "fineweb_edu_train.bin")
VAL_BIN = os.path.join(CORPUS_DIR, "fineweb_edu_val.bin")
PROGRESS_FILE = os.path.join(CORPUS_DIR, ".fineweb_edu_progress.json")
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DOC_SEP = b"\x00"
DEFAULT_VAL_FRAC = 0.005   # 0.5% to val
DEFAULT_MAX_DOCS = 0       # 0 = no cap
DEFAULT_VAL_DOCS = 10000


# ------------------------------------------------------------------------------------
# Functions

def load_progress():
    if not os.path.isfile(PROGRESS_FILE):
        return {"docs_written": 0, "train_bytes": 0, "val_bytes": 0, "val_docs_taken": 0}
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(p):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Download FineWeb-Edu sample-10BT and build byte-level .bin pair.")
    ap.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                    help="Stop after N documents (0 = no cap, full sample-10BT).")
    ap.add_argument("--val-docs", type=int, default=DEFAULT_VAL_DOCS,
                    help="Number of documents to reserve for the val .bin.")
    ap.add_argument("--restart", action="store_true",
                    help="Delete existing .bin and progress file, start fresh.")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run:", file=sys.stderr)
        print("  py -m pip install datasets", file=sys.stderr)
        sys.exit(1)

    os.makedirs(CORPUS_DIR, exist_ok=True)

    if args.restart:
        for path in (TRAIN_BIN, VAL_BIN, PROGRESS_FILE):
            if os.path.isfile(path):
                os.remove(path)
                print(f"[restart] removed {path}")

    progress = load_progress()
    print(f"[start] docs_already_written={progress['docs_written']:,}")
    print(f"[start] train_bytes={progress['train_bytes']:,}  val_bytes={progress['val_bytes']:,}")
    print(f"[start] streaming {DATASET_NAME} ({DATASET_CONFIG})...", flush=True)

    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    train_fh = open(TRAIN_BIN, "ab")
    val_fh = open(VAL_BIN, "ab")

    t0 = time.time()
    docs_seen = 0
    skip_until = progress["docs_written"]

    try:
        for doc in ds:
            docs_seen += 1
            if docs_seen <= skip_until:
                continue
            text = doc.get("text", "")
            if not text:
                continue
            data = text.encode("utf-8", errors="replace") + DOC_SEP

            if progress["val_docs_taken"] < args.val_docs:
                val_fh.write(data)
                progress["val_bytes"] += len(data)
                progress["val_docs_taken"] += 1
            else:
                train_fh.write(data)
                progress["train_bytes"] += len(data)

            progress["docs_written"] = docs_seen

            if docs_seen % 1000 == 0:
                train_fh.flush()
                val_fh.flush()
                save_progress(progress)
                rate = docs_seen / max(1e-6, time.time() - t0)
                tb = progress["train_bytes"] / (1024 ** 3)
                vb = progress["val_bytes"] / (1024 ** 3)
                print(f"[+] docs={docs_seen:,}  train={tb:.2f} GB  val={vb:.3f} GB  rate={rate:.0f} docs/s", flush=True)

            if args.max_docs > 0 and docs_seen >= args.max_docs:
                print(f"[done] reached --max-docs={args.max_docs}", flush=True)
                break
    finally:
        train_fh.close()
        val_fh.close()
        save_progress(progress)

    print()
    print(f"[final] docs_written={progress['docs_written']:,}")
    print(f"[final] train.bin = {progress['train_bytes']:,} bytes ({progress['train_bytes']/(1024**3):.2f} GB)")
    print(f"[final] val.bin   = {progress['val_bytes']:,} bytes ({progress['val_bytes']/(1024**3):.3f} GB)")
    print(f"[final] elapsed   = {time.time()-t0:.0f}s")
    print()
    print(f"  train -> {TRAIN_BIN}")
    print(f"  val   -> {VAL_BIN}")


if __name__ == "__main__":
    main()
