from pydantic import BaseModel


class BinaryMaskDetails(BaseModel):
    """Metadata used to explain the binary mask result."""

    title: str
    description: str
    average_filtering_time_ms: float
    timing_note: str
    width: int
    height: int
    pixel_count: int


class GradCamDetails(BaseModel):
    """Metadata used to explain the segmentation Grad-CAM-style result."""

    title: str
    description: str
    average_filtering_time_ms: float
    timing_note: str
    width: int
    height: int
    pixel_count: int


class SegmentationResponse(BaseModel):
    """Response model for the segmentation endpoint."""
    original_image: str      # base64-encoded PNG of the original (grayscale) image
    segmented_image: str     # base64-encoded PNG of the fuzzy-segmented image
    binary_image: str        # base64-encoded PNG of the Otsu-thresholded binary image
    gradcam_overlay_image: str   # base64-encoded PNG of the segmentation attention overlay
    gradcam_banded_image: str    # base64-encoded PNG of the three-band attention map
    masked_image: str        # base64-encoded PNG of the lesion extracted (bg removed)
    binary_details: BinaryMaskDetails
    gradcam_details: GradCamDetails
    is_normal: bool = False  # True when the classifier identifies the image as Normal skin
    message: str = "Segmentation completed successfully"


class ErrorResponse(BaseModel):
    """Generic error response model."""
    detail: str
