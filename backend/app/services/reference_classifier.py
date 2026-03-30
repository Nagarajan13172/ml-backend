"""
Reference-based (embedding similarity) classifier.

Requires NO model training. Instead:
  1. Put a few reference images in backend/data/reference/<ClassName>/
  2. Run:  python scripts/build_reference_db.py
  3. Restart the backend — this service loads the DB and classifies
     new images by cosine similarity to the class centroids.

Falls back gracefully if the DB file is absent (returns None).
"""
import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

BACKEND_DIR = Path(__file__).resolve().parents[2]
DB_PATH     = BACKEND_DIR / "models" / "reference_db.npz"
IMG_SIZE    = 224


# ── Feature extractor (EfficientNetB0, no top, avg-pooled) ───────────────────
@lru_cache(maxsize=1)
def _get_feature_extractor():
    try:
        import tensorflow as tf
    except ImportError:
        return None

    model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    model.trainable = False
    return model


@lru_cache(maxsize=1)
def _load_db():
    """Load precomputed class centroids. Returns None when DB is absent."""
    if not DB_PATH.exists():
        return None
    data = np.load(DB_PATH, allow_pickle=True)
    return {
        "centroids":   data["centroids"],    # (num_classes, dim)
        "class_names": data["class_names"].tolist(),
    }


def is_available() -> bool:
    """True when the reference DB exists and the feature extractor loads."""
    return DB_PATH.exists() and _get_feature_extractor() is not None


def clear_cache():
    _get_feature_extractor.cache_clear()
    _load_db.cache_clear()


def _preprocess(file_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            img = img.resize((IMG_SIZE, IMG_SIZE), resampling.BILINEAR)
            return np.asarray(img, dtype=np.float32)   # (H, W, 3)  range [0, 255]
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def embed_array(image_array: np.ndarray) -> np.ndarray:
    """
    Extract an EfficientNetB0 embedding from a (H, W, 3) float32 [0,255] array.
    Returns a 1-D feature vector.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow not installed.") from exc

    extractor = _get_feature_extractor()
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(
        image_array[np.newaxis, ...]
    )
    return extractor.predict(preprocessed, verbose=0)[0]


def classify(file_bytes: bytes) -> Optional[dict]:
    """
    Classify by cosine similarity to precomputed class centroids.

    Returns a dict compatible with ClassificationResponse, or None when the
    reference DB is not available (caller should fall back to the trained model).
    """
    db = _load_db()
    if db is None:
        return None

    img_array = _preprocess(file_bytes)
    embedding = embed_array(img_array)                    # (dim,)
    centroids = db["centroids"]                           # (C, dim)

    # Cosine similarity
    norm_emb = embedding  / (np.linalg.norm(embedding) + 1e-8)
    norm_cen = centroids  / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    similarities = norm_cen @ norm_emb                    # (C,)

    # Softmax to get probability-like confidence
    exp_sim      = np.exp(similarities - similarities.max())
    probabilities = exp_sim / exp_sim.sum()

    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_label": db["class_names"][predicted_index],
        "confidence":      float(probabilities[predicted_index]),
        "class_index":     predicted_index,
    }
