#!/usr/bin/env python3
"""
Download the Monkeypox Skin Lesion dataset from Kaggle into backend/data/.

Prerequisite:
    pip install kaggle
    Place your kaggle.json at ~/.kaggle/kaggle.json  (from kaggle.com → Account → API)

Usage:
    cd backend
    python scripts/download_dataset.py

Dataset: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20
Classes: Monkeypox, Chickenpox, Measles, Normal (and more)
"""
import sys
import shutil
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR    = BACKEND_DIR / "data"

# Kaggle dataset slug
DATASET = "joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"

# Map dataset folder names → our class names
# Adjust this mapping if the downloaded folder names differ
CLASS_MAP = {
    "Monkeypox":  "Monkeypox",
    "Chickenpox": "Chickenpox",
    "Measles":    "Measles",
    "Normal":     "Normal",
    # Some versions use alternate names:
    "HFMD":       None,   # skip
    "Cowpox":     None,   # skip
    "Healthy":    "Normal",
}

TARGET_CLASSES = {"Monkeypox", "Chickenpox", "Measles", "Normal"}


def check_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("  pip install kaggle")
        sys.exit(1)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Move the downloaded kaggle.json to ~/.kaggle/")
        sys.exit(1)


def download_and_extract():
    import kaggle

    download_dir = DATA_DIR / "_raw"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET} ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=str(download_dir), unzip=False, quiet=False)

    # Find the downloaded zip
    zips = list(download_dir.glob("*.zip"))
    if not zips:
        print("ERROR: No zip file found after download.")
        sys.exit(1)

    print(f"Extracting {zips[0].name} ...")
    with zipfile.ZipFile(zips[0], "r") as zf:
        zf.extractall(download_dir)

    return download_dir


def organise(raw_dir: Path):
    """Walk the extracted directory and copy images into train/ClassName/ layout."""
    train_dir = DATA_DIR / "train"
    val_dir   = DATA_DIR / "val"

    for target in TARGET_CLASSES:
        (train_dir / target).mkdir(parents=True, exist_ok=True)

    # Try to find existing train/val splits inside the download
    moved = 0
    for split in ("train", "val", "test", "Train", "Val", "Test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            # Also search one level deeper
            found = list(raw_dir.rglob(split))
            split_dir = found[0] if found else Path("/nonexistent")

        if split_dir.exists():
            dest_split = val_dir if split.lower() in ("val", "test") else train_dir
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                mapped = CLASS_MAP.get(cls_dir.name, cls_dir.name)
                if mapped not in TARGET_CLASSES:
                    continue
                out = dest_split / mapped
                out.mkdir(parents=True, exist_ok=True)
                for img in cls_dir.iterdir():
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        shutil.copy2(img, out / img.name)
                        moved += 1

    if moved == 0:
        # Flat structure — just copy everything by folder name
        for img_path in raw_dir.rglob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            cls_name = img_path.parent.name
            mapped   = CLASS_MAP.get(cls_name, cls_name)
            if mapped not in TARGET_CLASSES:
                continue
            out = train_dir / mapped
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out / img_path.name)
            moved += 1

    print(f"\nOrganised {moved} images.")
    for cls in TARGET_CLASSES:
        n_train = len(list((train_dir / cls).glob("*"))) if (train_dir / cls).exists() else 0
        n_val   = len(list((val_dir   / cls).glob("*"))) if (val_dir   / cls).exists() else 0
        print(f"  {cls:15s}  train={n_train}  val={n_val}")

    if moved == 0:
        print("\nWARNING: No images were copied. Check the folder structure inside:")
        print(f"  {raw_dir}")
        print("and update CLASS_MAP in this script if needed.")
    else:
        print(f"\nDataset ready at: {DATA_DIR}")
        print("Now run:  python scripts/train_model.py")


if __name__ == "__main__":
    check_kaggle()
    raw_dir = download_and_extract()
    organise(raw_dir)
