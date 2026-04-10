#!/usr/bin/env python3
"""
Prepare the Monkeypox Images dataset for training.

Clears existing backend/data/train and backend/data/val,
then copies images from the source folder with an 80/20 stratified split.

Usage:
    cd backend
    python scripts/prepare_data.py
    python scripts/prepare_data.py --source "../../Monkeypox Images" --val-split 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SRC = BACKEND_DIR.parent / "Monkeypox Images"
TRAIN_DIR   = BACKEND_DIR / "data" / "train"
VAL_DIR     = BACKEND_DIR / "data" / "val"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def prepare(source: Path, val_split: float, seed: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    classes = sorted(p.name for p in source.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No class subdirectories found in {source}")

    print(f"Source  : {source}")
    print(f"Classes : {classes}")
    print(f"Val split: {val_split*100:.0f}%\n")

    # ── Wipe old splits ───────────────────────────────────────────────────────
    for split_dir in (TRAIN_DIR, VAL_DIR):
        if split_dir.exists():
            shutil.rmtree(split_dir)
            print(f"Removed  : {split_dir}")

    random.seed(seed)
    totals = {"train": 0, "val": 0}

    for cls in classes:
        src_cls = source / cls
        images  = [f for f in src_cls.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if not images:
            print(f"  [{cls}] SKIPPED — no images found")
            continue

        random.shuffle(images)
        n_val   = max(1, int(len(images) * val_split))
        n_train = len(images) - n_val

        splits = [
            (TRAIN_DIR / cls, images[:n_train]),
            (VAL_DIR   / cls, images[n_train:]),
        ]
        for dest_dir, files in splits:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)

        totals["train"] += n_train
        totals["val"]   += n_val
        print(f"  [{cls}]  train={n_train}  val={n_val}  (total={len(images)})")

    print(f"\nDone.  Total → train={totals['train']}  val={totals['val']}")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Val   dir: {VAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SRC,
        help="Path to the root folder containing class subdirectories",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    prepare(args.source, args.val_split, args.seed)
