from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.segmentation import SegmentationResponse
from app.services import segmentation_service

router = APIRouter(
    prefix="/segment",
    tags=["Segmentation"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}


@router.post(
    "",
    response_model=SegmentationResponse,
    summary="Fuzzy Image Segmentation",
    description=(
        "Upload an image file (PNG, JPEG, BMP, TIFF) and receive three processed images:\n"
        "- **original_image**: grayscale version of your input\n"
        "- **segmented_image**: fuzzy-membership-segmented image\n"
        "- **binary_image**: Otsu-thresholded binary image\n\n"
        "All images are returned as **base64-encoded PNG** strings."
    ),
)
async def segment_image(file: UploadFile = File(..., description="Image file to segment")):
    """
    Run fuzzy segmentation + Otsu thresholding on the uploaded image.
    """
    # Validate content type
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

        result = segmentation_service.process_image(file_bytes)

        return SegmentationResponse(
            original_image=result["original_image"],
            segmented_image=result["segmented_image"],
            binary_image=result["binary_image"],
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
