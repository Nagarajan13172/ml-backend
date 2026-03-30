"""
Zero-shot image classification using OpenAI CLIP.

No training data or reference images required.
The model compares the uploaded image against text descriptions
of each disease class and returns the best match.
"""
import io
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# ── Skin lesion labels with dermoscopy-specific descriptions ──────────────────
CLASS_DESCRIPTIONS = {
    "Actinic keratosis": [
        "actinic keratosis rough scaly patch on sun-damaged skin dermoscopy",
        "solar keratosis precancerous lesion with crusty surface skin",
        "actinic keratosis pink scaly skin lesion with keratin crust",
    ],
    "Atopic Dermatitis": [
        "atopic dermatitis eczema inflamed itchy red skin rash",
        "eczema skin with red patches dry flaky skin and inflammation",
        "atopic dermatitis chronic skin condition with red irritated patches",
    ],
    "Benign keratosis": [
        "seborrheic keratosis benign warty stuck-on pigmented skin growth",
        "benign keratosis brown waxy rough skin lesion dermoscopy",
        "seborrheic wart benign pigmented keratosis on skin dermoscopy",
    ],
    "Dermatofibroma": [
        "dermatofibroma firm benign skin nodule with central white scar dermoscopy",
        "dermatofibroma pigmented firm small bump on skin dermoscopy",
        "benign dermatofibroma skin lesion with peripheral pigment ring dermoscopy",
    ],
    "Melanocytic nevus": [
        "melanocytic nevus common mole pigmented skin lesion dermoscopy",
        "benign mole with regular borders and uniform pigmentation dermoscopy",
        "nevus melanocytic brown pigmented spot on skin dermoscopy",
    ],
    "Melanoma": [
        "melanoma malignant skin cancer irregular border asymmetric pigmented lesion dermoscopy",
        "melanoma dark irregular skin lesion with multiple colors dermoscopy",
        "malignant melanoma skin cancer with atypical pigment network dermoscopy",
    ],
    "Squamous cell carcinoma": [
        "squamous cell carcinoma skin cancer scaly red ulcerated lesion",
        "squamous cell carcinoma thick crusty skin lesion dermoscopy",
        "SCC skin malignancy with irregular surface and keratin dermoscopy",
    ],
    "Tinea Ringworm Candidiasis": [
        "tinea ringworm fungal infection circular ring-shaped skin rash",
        "ringworm tinea corporis circular scaly skin fungal infection",
        "candidiasis fungal skin infection red itchy rash dermoscopy",
    ],
    "Vascular lesion": [
        "vascular skin lesion red purple angioma blood vessel growth dermoscopy",
        "cherry angioma hemangioma red vascular skin lesion dermoscopy",
        "vascular lesion bright red lacunae blood vessel dermoscopy",
    ],
}

# Ordered list matching the API's class_names in .env
CLASS_NAMES = list(CLASS_DESCRIPTIONS.keys())


@lru_cache(maxsize=1)
def _load_clip():
    """Load CLIP model and processor (cached after first load)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers not installed. Run: pip install transformers torch"
        ) from exc

    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def is_available() -> bool:
    """Return True if CLIP can be loaded (transformers + torch installed)."""
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


def clear_cache():
    _load_clip.cache_clear()


def _preprocess_image(file_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            return ImageOps.exif_transpose(img).convert("RGB").copy()
    except Exception as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def classify(file_bytes: bytes) -> Optional[dict]:
    """
    Zero-shot classify an image using CLIP.

    Returns a dict with predicted_label, confidence, class_index,
    or None if CLIP is not available.
    """
    if not is_available():
        return None

    import torch

    model, processor = _load_clip()
    pil_image = _preprocess_image(file_bytes)

    # Build a flat list of all descriptions with their class index
    all_texts  = []
    class_idx_for_text = []
    for idx, (cls, descriptions) in enumerate(CLASS_DESCRIPTIONS.items()):
        for desc in descriptions:
            all_texts.append(desc)
            class_idx_for_text.append(idx)

    inputs = processor(
        text=all_texts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs   = model(**inputs)
        logits    = outputs.logits_per_image.squeeze(0)   # (num_texts,)
        text_probs = logits.softmax(dim=0).numpy()         # (num_texts,)

    # Aggregate probabilities per class (max pooling over descriptions)
    num_classes   = len(CLASS_NAMES)
    class_scores  = np.zeros(num_classes, dtype=np.float32)
    for text_i, cls_i in enumerate(class_idx_for_text):
        class_scores[cls_i] = max(class_scores[cls_i], float(text_probs[text_i]))

    # Re-normalise to sum to 1
    class_scores  = class_scores / (class_scores.sum() + 1e-8)
    predicted_idx = int(np.argmax(class_scores))

    return {
        "predicted_label": CLASS_NAMES[predicted_idx],
        "confidence":      float(class_scores[predicted_idx]),
        "class_index":     predicted_idx,
    }
