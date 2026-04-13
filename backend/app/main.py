import asyncio
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import classification, segmentation

logger = logging.getLogger(__name__)


def _warmup_classifier():
    """Pre-load model and run one dummy inference to JIT-compile TF graph."""
    from app.services.classification_service import (
        ClassifierNotReadyError,
        _load_classifier_bundle,
    )
    try:
        bundle = _load_classifier_bundle()
        h, w = bundle.image_size
        dummy = np.zeros((1, h, w, 3), dtype=np.float32)
        bundle.model.predict(dummy, verbose=0)
        logger.info("Classifier warmed up — model ready at '%s'", bundle.model_path)
    except ClassifierNotReadyError as exc:
        logger.warning("Classifier not available at startup: %s", exc)
    except Exception as exc:
        logger.warning("Classifier warmup failed (will retry on first request): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run warmup in background so the server is immediately ready to accept requests.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _warmup_classifier)
    yield


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A FastAPI service for monkeypox-related image workflows. "
        "It supports lightweight lesion **classification** for top-label prediction "
        "and image **segmentation** for diagnostic visual outputs."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware — allow frontend (React/Vue/etc.) to call this API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(segmentation.router, prefix="/api")
app.include_router(classification.router, prefix="/api")


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}",
        "docs": "/docs",
        "classification_health": "/api/classify/health",
        "health": "/api/segment/health",
    }
