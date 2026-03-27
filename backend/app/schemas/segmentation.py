from pydantic import BaseModel


class SegmentationResponse(BaseModel):
    """Response model for the segmentation endpoint."""
    original_image: str      # base64-encoded PNG of the original (grayscale) image
    segmented_image: str     # base64-encoded PNG of the fuzzy-segmented image
    binary_image: str        # base64-encoded PNG of the Otsu-thresholded binary image
    message: str = "Segmentation completed successfully"


class ErrorResponse(BaseModel):
    """Generic error response model."""
    detail: str
