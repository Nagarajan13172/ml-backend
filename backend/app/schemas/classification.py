from typing import Optional

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    """Response model for the image classification endpoint."""

    predicted_label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence for the predicted class.")
    class_index: int = Field(..., ge=0, description="Index of the predicted class.")
    message: str = "Classification completed successfully"


class ClassificationHealthResponse(BaseModel):
    """Health model for the classifier runtime."""

    status: str
    service: str
    model_ready: bool
    model_path: str
    detail: Optional[str] = None
