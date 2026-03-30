#!/usr/bin/env python3
"""
Build a reference embedding database from a small set of labeled images.

── Setup ─────────────────────────────────────────────────────────────────────
Put at least ONE image per class here (more = better):

  backend/data/reference/
    Monkeypox/
      mpox1.jpg  mpox2.jpg  …
    Chickenpox/
      cp1.jpg  cp2.jpg  …
    Measles/
      measles1.jpg  …
    Normal/
      normal1.jpg  …

── Usage ─────────────────────────────────────────────────────────────────────
    cd backend
    python scripts/build_reference_db.py

The script saves  backend/models/reference_db.npz.
Restart the API server — it will automatically use the reference classifier.
──────────────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR  = Path(__file__).resolve().parents[1]
REFERENCE_DIR = BACKEND_DIR / "data" / "reference"
DB_PATH       = BACKEND_DIR / "models" / "reference_db.npz"

CLASS_NAMES   = ["Monkeypox", "Chickenpox", "Measles", "Normal"]
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def main():
    # ── validate reference directory ──────────────────────────────────────
    if not REFERENCE_DIR.exists():
        print(f"ERROR: Reference directory not found: {REFERENCE_DIR}")
        print()
        print("Create it with at least one image per class:")
        for cls in CLASS_NAMES:
            print(f"  {REFERENCE_DIR / cls}/")
        sys.exit(1)

    # ── import tensorflow ─────────────────────────────────────────────────
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed.")
        sys.exit(1)

    from PIL import Image, ImageOps

    print(f"TensorFlow {tf.__version__}")
    IMG_SIZE = 224

    # ── load EfficientNetB0 feature extractor ─────────────────────────────
    print("Loading EfficientNetB0 (ImageNet weights)…")
    extractor = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    extractor.trainable = False

    def embed(image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            img = img.resize((IMG_SIZE, IMG_SIZE), resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
        preprocessed = tf.keras.applications.efficientnet.preprocess_input(arr[np.newaxis, ...])
        return extractor.predict(preprocessed, verbose=0)[0]

    # ── compute per-class centroids ───────────────────────────────────────
    centroids   = []
    valid_names = []
    missing     = []

    for cls in CLASS_NAMES:
        cls_dir = REFERENCE_DIR / cls
        images  = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS] if cls_dir.exists() else []

        if not images:
            missing.append(cls)
            print(f"  SKIP  {cls:15s} — no images found in {cls_dir}")
            continue

        embeddings = []
        for img_path in images:
            try:
                emb = embed(img_path)
                embeddings.append(emb)
                print(f"  OK    {cls:15s}  {img_path.name}")
            except Exception as exc:
                print(f"  WARN  {cls:15s}  {img_path.name}: {exc}")

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            centroids.append(centroid)
            valid_names.append(cls)

    if not centroids:
        print("\nERROR: No embeddings computed. Add images to the reference directory.")
        sys.exit(1)

    if missing:
        print(f"\nWARNING: Missing classes: {missing}")
        print("The classifier will only predict from available classes:", valid_names)

    # ── save ─────────────────────────────────────────────────────────────
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DB_PATH,
        centroids=np.stack(centroids),
        class_names=np.array(valid_names),
    )

    print(f"\nReference DB saved → {DB_PATH}")
    print(f"  Classes : {valid_names}")
    print(f"  Embedding dim : {centroids[0].shape[0]}")
    print("\nRestart the backend server — it will now use reference-based classification.")


if __name__ == "__main__":
    main()
