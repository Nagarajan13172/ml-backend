import asyncio
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncGenerator

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import settings


class ClassifierNotReadyError(RuntimeError):
    """Raised when the classifier cannot be used yet."""


@dataclass(frozen=True)
class ClassifierBundle:
    model: Any
    class_names: list[str]
    image_size: tuple[int, int]
    model_path: str


def _import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ClassifierNotReadyError(
            "TensorFlow is not installed. Add it to the backend environment to enable classification."
        ) from exc

    return tf


def _build_fuzzy_membership_layer(tf):
    class FuzzyTriangularMembership(tf.keras.layers.Layer):
        def __init__(self, a=0.2, b=0.6, c=1.0, **kwargs):
            super().__init__(**kwargs)
            self.a = float(a)
            self.b = float(b)
            self.c = float(c)

        def call(self, inputs):
            left = (inputs - self.a) / max(self.b - self.a, 1e-6)
            right = (self.c - inputs) / max(self.c - self.b, 1e-6)
            return tf.clip_by_value(tf.minimum(left, right), 0.0, 1.0)

        def get_config(self):
            config = super().get_config()
            config.update({"a": self.a, "b": self.b, "c": self.c})
            return config

    return FuzzyTriangularMembership


def _resolve_class_names(output_size: int) -> list[str]:
    configured_names = settings.classification_class_names
    if not configured_names:
        return [f"Class {index}" for index in range(output_size)]

    if len(configured_names) < output_size:
        configured_names = configured_names + [
            f"Class {index}" for index in range(len(configured_names), output_size)
        ]

    return configured_names[:output_size]


@lru_cache(maxsize=1)
def _load_classifier_bundle() -> ClassifierBundle:
    tf = _import_tensorflow()
    model_path = settings.classification_model_path

    if not model_path.exists():
        raise ClassifierNotReadyError(
            f"Classifier model file was not found at '{model_path}'. "
            "Save your trained model there or set CLASSIFICATION_MODEL_PATH."
        )

    fuzzy_layer = _build_fuzzy_membership_layer(tf)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"FuzzyTriangularMembership": fuzzy_layer},
        compile=False,
    )

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if not output_shape or output_shape[-1] is None:
        raise ClassifierNotReadyError(
            f"Unable to determine classifier output shape for model '{model_path}'."
        )

    output_size = int(output_shape[-1])
    return ClassifierBundle(
        model=model,
        class_names=_resolve_class_names(output_size),
        image_size=(settings.CLASSIFICATION_IMAGE_SIZE, settings.CLASSIFICATION_IMAGE_SIZE),
        model_path=str(model_path),
    )


def clear_classifier_cache() -> None:
    """Clear cached model state. Useful in tests and during hot reload."""
    _load_classifier_bundle.cache_clear()


def _prepare_image(file_bytes: bytes, image_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize(image_size, resampling.BILINEAR)
            # Keep [0, 255] — the model's first layer (Rescaling 1/255) handles normalisation
            image_array = np.asarray(image, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    return np.expand_dims(image_array, axis=0)


def _augment_image(image: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a single (H, W, 3) float32 [0,255] image."""
    img = image.copy()
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :]
    # Random vertical flip
    if np.random.rand() > 0.5:
        img = img[::-1, :, :]
    # Random brightness ±15%  (scale in [0,255] space)
    img = np.clip(img * np.random.uniform(0.85, 1.15), 0.0, 255.0).astype(np.float32)
    # Random channel shift ±5% of 255
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] + np.random.uniform(-12.75, 12.75), 0.0, 255.0)
    return img.astype(np.float32)


def classify_image(file_bytes: bytes) -> dict:
    """
    Classify a single uploaded image.

    Priority:
      1. Trained Keras model  — most accurate when a proper model exists
      2. Reference DB         — embedding similarity fallback
      3. CLIP zero-shot       — last resort when no model or DB is available
    """
    from app.services import clip_classifier, reference_classifier

    # 1 ── Trained Keras model (primary)
    try:
        classifier   = _load_classifier_bundle()
        image_batch  = _prepare_image(file_bytes, classifier.image_size)
        predictions  = classifier.model.predict(image_batch, verbose=0)
        scores       = np.asarray(predictions, dtype=np.float32).squeeze()
        if scores.ndim == 1 and scores.size > 0:
            predicted_index = int(np.argmax(scores))
            return {
                "predicted_label": classifier.class_names[predicted_index],
                "confidence":      float(scores[predicted_index]),
                "class_index":     predicted_index,
            }
    except ClassifierNotReadyError:
        pass  # fall through

    # 2 ── Reference embedding similarity
    result = reference_classifier.classify(file_bytes)
    if result is not None:
        return result

    # 3 ── CLIP zero-shot (last resort)
    result = clip_classifier.classify(file_bytes)
    if result is not None:
        return result

    raise ClassifierNotReadyError(
        "No classifier is available. Train a model or build a reference DB."
    )


async def tta_stream(file_bytes: bytes, total_epochs: int = 100) -> AsyncGenerator[dict, None]:
    """
    Test-Time Augmentation stream: 100 augmented inference passes.

    Priority:
      1. Keras model TTA   (most accurate when trained model exists)
      2. Reference embedding TTA
      3. CLIP zero-shot TTA (last resort)
    """
    from app.services import clip_classifier as clip_cls
    from app.services import reference_classifier as ref_cls

    loop = asyncio.get_event_loop()

    # ── Keras model TTA (primary) ─────────────────────────────────────────
    keras_ready = True
    try:
        classifier = await loop.run_in_executor(None, _load_classifier_bundle)
    except ClassifierNotReadyError:
        keras_ready = False

    if keras_ready:
        try:
            base_image = _prepare_image(file_bytes, classifier.image_size)[0]  # (H,W,3) [0,255]
        except ValueError as exc:
            yield {"error": str(exc)}
            return

        batch_size  = 10
        num_batches = total_epochs // batch_size
        all_scores: list[np.ndarray] = []
        epoch = 0

        for _ in range(num_batches):
            augmented_batch = np.stack([_augment_image(base_image) for _ in range(batch_size)])
            preds = await loop.run_in_executor(
                None,
                lambda b=augmented_batch: classifier.model.predict(b, verbose=0),
            )
            batch_scores = np.asarray(preds, dtype=np.float32)

            for i in range(batch_size):
                epoch += 1
                all_scores.append(batch_scores[i])
                avg = np.mean(all_scores, axis=0)
                idx = int(np.argmax(avg))
                yield {
                    "epoch":           epoch,
                    "done":            epoch == total_epochs,
                    "predicted_label": classifier.class_names[idx],
                    "confidence":      float(avg[idx]),
                    "class_index":     idx,
                }
                await asyncio.sleep(0.05)
        return

    # ── CLIP TTA (fallback when no trained model) ─────────────────────────
    if clip_cls.is_available():
        import torch
        from PIL import Image as PILImage

        try:
            model, processor = await loop.run_in_executor(None, clip_cls._load_clip)
            pil_image = clip_cls._preprocess_image(file_bytes)
        except (ValueError, RuntimeError) as exc:
            yield {"error": str(exc)}
            return

        img_array = np.asarray(pil_image, dtype=np.float32)  # (H, W, 3)

        # Build text descriptions once
        all_texts, class_idx_for_text = [], []
        for idx, (_, descs) in enumerate(clip_cls.CLASS_DESCRIPTIONS.items()):
            for d in descs:
                all_texts.append(d)
                class_idx_for_text.append(idx)
        num_classes = len(clip_cls.CLASS_NAMES)

        def _clip_augmented(arr: np.ndarray) -> np.ndarray:
            """Augment (H,W,3 float32 [0,255]), run CLIP, return per-class scores."""
            aug_arr = _augment_image(arr / 255.0) * 255.0   # augment then rescale
            aug_img = PILImage.fromarray(aug_arr.astype(np.uint8))
            inputs  = processor(
                text=all_texts, images=aug_img,
                return_tensors="pt", padding=True, truncation=True,
            )
            with torch.no_grad():
                logits = model(**inputs).logits_per_image.squeeze(0)
                probs  = logits.softmax(dim=0).numpy()

            scores = np.zeros(num_classes, dtype=np.float32)
            for ti, ci in enumerate(class_idx_for_text):
                scores[ci] = max(scores[ci], float(probs[ti]))
            return scores / (scores.sum() + 1e-8)

        accumulated = np.zeros(num_classes, dtype=np.float32)
        for epoch in range(1, total_epochs + 1):
            epoch_scores = await loop.run_in_executor(None, _clip_augmented, img_array)
            accumulated  = accumulated + epoch_scores
            avg_scores   = accumulated / epoch
            idx          = int(np.argmax(avg_scores))
            yield {
                "epoch":           epoch,
                "done":            epoch == total_epochs,
                "predicted_label": clip_cls.CLASS_NAMES[idx],
                "confidence":      float(avg_scores[idx]),
                "class_index":     idx,
            }
            await asyncio.sleep(0.05)
        return

    # ── Reference-based TTA (preferred when DB exists) ────────────────────
    if ref_cls.is_available():
        extractor = await loop.run_in_executor(None, ref_cls._get_feature_extractor)

        try:
            img_array = ref_cls._preprocess(file_bytes)   # (H, W, 3) float32 [0,255]
        except ValueError as exc:
            yield {"error": str(exc)}
            return

        def _embed_augmented(arr: np.ndarray) -> np.ndarray:
            """Augment (H,W,3 float32 [0,1]) → embed."""
            aug = _augment_image(arr)   # augment in [0,255] space
            try:
                import tensorflow as tf
                preprocessed = tf.keras.applications.efficientnet.preprocess_input(
                    aug[np.newaxis, ...]
                )
                return extractor.predict(preprocessed, verbose=0)[0]
            except Exception:
                return ref_cls.embed_array(aug)

        # Pre-load DB centroids once
        db = ref_cls._load_db()
        centroids  = db["centroids"]
        class_names = db["class_names"]
        norm_cen = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

        all_sims: list[np.ndarray] = []
        for epoch in range(1, total_epochs + 1):
            emb = await loop.run_in_executor(None, _embed_augmented, img_array)
            norm_emb = emb / (np.linalg.norm(emb) + 1e-8)
            sims = norm_cen @ norm_emb
            all_sims.append(sims)

            avg_sims = np.mean(all_sims, axis=0)
            exp_s = np.exp(avg_sims - avg_sims.max())
            probs = exp_s / exp_s.sum()
            idx = int(np.argmax(probs))

            yield {
                "epoch":           epoch,
                "done":            epoch == total_epochs,
                "predicted_label": class_names[idx],
                "confidence":      float(probs[idx]),
                "class_index":     idx,
            }
            await asyncio.sleep(0.05)
        return

    # ── Keras TTA (fallback) ──────────────────────────────────────────────
    try:
        classifier = await loop.run_in_executor(None, _load_classifier_bundle)
    except ClassifierNotReadyError as exc:
        yield {"error": str(exc)}
        return

    try:
        base_image = _prepare_image(file_bytes, classifier.image_size)[0]  # (H, W, 3)
    except ValueError as exc:
        yield {"error": str(exc)}
        return

    batch_size = 10
    num_batches = total_epochs // batch_size  # 10 batches
    all_scores: list[np.ndarray] = []
    epoch = 0

    for _ in range(num_batches):
        # Build batch of augmented images
        augmented_batch = np.stack(
            [_augment_image(base_image) for _ in range(batch_size)]
        )  # (10, H, W, 3)

        # Predict in thread pool to avoid blocking the event loop
        preds = await loop.run_in_executor(
            None,
            lambda b=augmented_batch: classifier.model.predict(b, verbose=0),
        )
        batch_scores = np.asarray(preds, dtype=np.float32)  # (10, num_classes)

        # Emit one event per image in the batch
        for i in range(batch_size):
            epoch += 1
            all_scores.append(batch_scores[i])
            avg = np.mean(all_scores, axis=0)
            idx = int(np.argmax(avg))

            yield {
                "epoch": epoch,
                "done": epoch == total_epochs,
                "predicted_label": classifier.class_names[idx],
                "confidence": float(avg[idx]),
                "class_index": idx,
            }
            # Small delay for smooth frontend animation (50 ms per epoch)
            await asyncio.sleep(0.05)


def classifier_health() -> dict:
    """Return readiness information for the classification runtime."""
    model_path = str(settings.classification_model_path)

    try:
        _load_classifier_bundle()
    except ClassifierNotReadyError as exc:
        return {
            "status": "not_ready",
            "service": "classification",
            "model_ready": False,
            "model_path": model_path,
            "detail": str(exc),
        }

    return {
        "status": "ok",
        "service": "classification",
        "model_ready": True,
        "model_path": model_path,
        "detail": None,
    }
