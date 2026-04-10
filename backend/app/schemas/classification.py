from typing import Optional

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    """Response model for the image classification endpoint."""

    predicted_label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence for the predicted class.")
    class_index: int = Field(..., ge=0, description="Index of the predicted class.")
    message: str = "Classification completed successfully"


class ClassificationGradCAMResponse(BaseModel):
    """Response model for the Grad-CAM classification endpoint."""

    predicted_label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence for the predicted class.")
    class_index: int = Field(..., ge=0, description="Index of the predicted class.")
    message: str = "Classification with Grad-CAM completed successfully"
    gradcam_heatmap_image: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded PNG of the raw Grad-CAM heatmap using the jet colormap. "
            "Blue = low importance, red = high importance."
        ),
    )
    gradcam_overlay_image: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded PNG of the Grad-CAM heatmap blended onto the original image, "
            "showing which regions most influenced the classification decision."
        ),
    )
    gradcam_available: bool = Field(
        False,
        description="True when Grad-CAM was successfully computed for the Keras model.",
    )


class ClassificationHealthResponse(BaseModel):
    """Health model for the classifier runtime."""

    status: str
    service: str
    model_ready: bool
    model_path: str
    detail: Optional[str] = None
