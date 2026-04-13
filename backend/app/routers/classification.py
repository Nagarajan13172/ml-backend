import asyncio
import json
from functools import partial

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.classification import (
    ClassificationGradCAMResponse,
    ClassificationHealthResponse,
    ClassificationResponse,
)
from app.services import classification_service


router = APIRouter(
    prefix="/classify",
    tags=["Classification"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}


@router.post(
    "",
    response_model=ClassificationResponse,
    summary="Classify an uploaded lesion image",
    description=(
        "Upload a lesion image and receive only the top predicted class label "
        "plus its confidence score."
    ),
)
async def classify_uploaded_image(file: UploadFile = File(..., description="Image file to classify")):
    """Run trained image classification on the uploaded image."""
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
        result = await loop.run_in_executor(
            None, partial(classification_service.classify_image, file_bytes)
        )
        return ClassificationResponse(**result)

    except HTTPException:
        raise
    except classification_service.ClassifierNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image classification failed: {str(exc)}",
        ) from exc


@router.post(
    "/gradcam",
    response_model=ClassificationGradCAMResponse,
    summary="Classify image and generate Grad-CAM visualisation",
    description=(
        "Classifies the uploaded lesion image using the trained EfficientNetB0 model "
        "and returns two Grad-CAM visualisations:\n\n"
        "- **gradcam_heatmap_image** — standalone jet-colourmap heatmap "
        "(blue = low importance → red = high importance).\n"
        "- **gradcam_overlay_image** — heatmap blended onto the original image, "
        "highlighting the regions that drove the classification decision.\n\n"
        "Both images are base64-encoded PNGs. `gradcam_available` is `false` when "
        "the trained Keras model is not loaded (fallback classifiers do not support Grad-CAM)."
    ),
)
async def classify_with_gradcam(
    file: UploadFile = File(..., description="Lesion image to classify"),
):
    """Classify and produce Grad-CAM heatmap + overlay."""
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
        result = await loop.run_in_executor(
            None, classification_service.classify_with_gradcam, file_bytes
        )
        return ClassificationGradCAMResponse(**result)

    except HTTPException:
        raise
    except classification_service.ClassifierNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM classification failed: {str(exc)}",
        ) from exc


@router.post(
    "/stream",
    summary="Classify via 100-epoch Test-Time Augmentation (SSE)",
    description=(
        "Streams Server-Sent Events as 100 augmented inference passes run. "
        "Each event carries `epoch`, `predicted_label`, `confidence`, and `done`. "
        "The final event (`done: true`) contains the stable averaged prediction."
    ),
)
async def classify_stream(file: UploadFile = File(..., description="Image file to classify")):
    """Run 100 TTA epochs and stream progress via SSE."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    async def event_generator():
        async for event in classification_service.tta_stream(file_bytes):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get(
    "/health",
    response_model=ClassificationHealthResponse,
    summary="Health check for the classification service",
    tags=["Health"],
)
async def health_check():
    """Return classifier readiness information."""
    return ClassificationHealthResponse(**classification_service.classifier_health())
