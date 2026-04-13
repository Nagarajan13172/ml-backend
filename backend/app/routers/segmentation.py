import asyncio
import base64
import io
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.schemas.segmentation import SegmentationResponse
from app.services import segmentation_service

router = APIRouter(
    prefix="/segment",
    tags=["Segmentation"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}

NORMAL_CLASS_NAMES = {"normal", "Normal", "NORMAL"}


def _blank_png(width: int = 224, height: int = 224) -> str:
    """Return a base64-encoded all-black PNG of the given size."""
    arr = np.zeros((height, width), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _grayscale_png(file_bytes: bytes) -> str:
    """Return a base64-encoded grayscale PNG from raw image bytes."""
    from skimage import color, io as skio

    img = skio.imread(io.BytesIO(file_bytes), as_gray=False)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    else:
        gray = np.clip(img, 0, 255).astype(np.uint8)

    h, w = gray.shape
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), h, w


def _normal_response(file_bytes: bytes) -> SegmentationResponse:
    """Build a clean SegmentationResponse for a Normal (lesion-free) image."""
    gray_b64, h, w = _grayscale_png(file_bytes)
    blank = _blank_png(w, h)

    placeholder_details = {
        "title": "No Lesion Detected",
        "description": "The classifier identified this image as Normal skin. No lesion segmentation was performed.",
        "average_filtering_time_ms": 0.0,
        "timing_note": "Skipped — Normal image.",
        "width": w,
        "height": h,
        "pixel_count": w * h,
    }

    return SegmentationResponse(
        original_image=gray_b64,
        segmented_image=blank,
        binary_image=blank,
        gradcam_overlay_image=blank,
        gradcam_banded_image=blank,
        masked_image=blank,
        binary_details=placeholder_details,
        gradcam_details=placeholder_details,
        is_normal=True,
        message="Normal skin detected — no lesion areas found.",
    )


def _try_classify_label(file_bytes: bytes) -> Optional[str]:
    """
    Try to classify the image using the trained Keras model.
    Returns the predicted label string, or None if the classifier is unavailable.
    """
    try:
        from app.services.classification_service import (
            ClassifierNotReadyError,
            classify_image,
        )
        result = classify_image(file_bytes)
        return result.get("predicted_label")
    except Exception:
        return None


@router.post(
    "",
    response_model=SegmentationResponse,
    summary="Fuzzy Image Segmentation",
    description=(
        "Upload an image file (PNG, JPEG, BMP, TIFF) and receive processed images plus binary-mask metadata:\n"
        "- **original_image**: grayscale version of your input\n"
        "- **segmented_image**: fuzzy-membership-segmented image\n"
        "- **binary_image**: hybrid adaptive/Otsu binary image\n"
        "- **gradcam_overlay_image**: three-band segmentation attention overlay\n"
        "- **gradcam_banded_image**: discrete green/yellow/red affected-area map\n"
        "- **masked_image**: transparent lesion cutout based on the binary mask\n"
        "- **binary_details**: description and timing metadata for the binary mask\n\n"
        "- **gradcam_details**: description and timing metadata for the segmentation attention map\n\n"
        "- **is_normal**: true when the classifier identifies the image as Normal (no lesion segmentation run)\n\n"
        "All images are returned as **base64-encoded PNG** strings."
    ),
)
async def segment_image(file: UploadFile = File(..., description="Image file to segment")):
    """
    Run fuzzy segmentation + hybrid adaptive/Otsu masking on the uploaded image.
    If the trained classifier identifies the image as Normal, returns clean empty
    masks instead of running segmentation.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        loop = asyncio.get_event_loop()

        # ── Normal-skin gate ─────────────────────────────────────────────────
        # Classify first (model is pre-warmed at startup). If the classifier
        # says "Normal", skip lesion segmentation entirely and return clean masks.
        label = await loop.run_in_executor(None, _try_classify_label, file_bytes)
        if label is not None and label in NORMAL_CLASS_NAMES:
            return await loop.run_in_executor(None, _normal_response, file_bytes)

        # ── Run full segmentation for diseased images ────────────────────────
        result = await loop.run_in_executor(
            None, segmentation_service.process_image, file_bytes
        )

        return SegmentationResponse(
            original_image=result["original_image"],
            segmented_image=result["segmented_image"],
            binary_image=result["binary_image"],
            gradcam_overlay_image=result["gradcam_overlay_image"],
            gradcam_banded_image=result["gradcam_banded_image"],
            masked_image=result["masked_image"],
            binary_details=result["binary_details"],
            gradcam_details=result["gradcam_details"],
            is_normal=False,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(exc)}",
        ) from exc


@router.get(
    "/health",
    summary="Health check for segmentation service",
    tags=["Health"],
)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "segmentation"}
