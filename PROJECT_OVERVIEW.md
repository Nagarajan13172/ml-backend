# Monkeypox AI Project - Overview

## Project Summary

**Monkeypox AI** is a full-stack machine learning application designed for automated lesion classification and image segmentation. The system helps diagnose skin conditions by analyzing medical images, with support for classifying:
- Monkeypox
- Chickenpox
- Measles
- Normal (healthy) skin

The project consists of two main components: a **FastAPI backend** serving ML models and a **React/Vite frontend** for user interaction.

---

## Project Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React/Vite Frontend                      │
│              (TypeScript + React Components)                 │
│   - Image Upload Interface                                   │
│   - Classification Results Display                           │
│   - Segmentation Visualization                               │
│   - Theme Switching (Light/Dark)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP Requests
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Server                    │
│                   (Python 3.x + FastAPI)                     │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              API Routes & Endpoints                     │  │
│  │  - Classification Router (/api/classify)               │  │
│  │  - Segmentation Router (/api/segment)                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Service Layer                             │  │
│  │  - classification_service.py                           │  │
│  │  - segmentation_service.py                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Machine Learning Models                       │  │
│  │  - TensorFlow/Keras Models (.keras files)              │  │
│  │  - CLIP-based Classification                           │  │
│  │  - Fuzzy Segmentation & Masking                        │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

### Backend (`/backend`)

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py                    # Configuration management via .env
│   ├── main.py                      # FastAPI app initialization
│   ├── routers/
│   │   ├── classification.py        # Classification endpoints
│   │   └── segmentation.py          # Segmentation endpoints
│   ├── schemas/
│   │   ├── classification.py        # Pydantic models for classification API
│   │   └── segmentation.py          # Pydantic models for segmentation API
│   └── services/
│       ├── classification_service.py # Core classification logic
│       ├── clip_classifier.py       # CLIP-based classifier
│       ├── reference_classifier.py  # Reference-based classifier
│       └── segmentation_service.py  # Core segmentation logic
├── data/
│   ├── reference/                   # Reference images for CLIP classifier
│   │   ├── Chickenpox/
│   │   └── Measles/
│   ├── train/                       # Training dataset
│   │   ├── Chickenpox/
│   │   ├── Measles/
│   │   ├── Monkeypox/
│   │   └── Normal/
│   └── val/                         # Validation dataset
│       ├── Chickenpox/
│       ├── Measles/
│       ├── Monkeypox/
│       └── Normal/
├── models/
│   ├── monkeypox_classifier.keras   # Primary trained model
│   ├── monkeypox_classifier_ft.keras # Fine-tuned variant
│   └── README.md
├── scripts/
│   ├── build_reference_db.py        # Generate CLIP reference embeddings
│   ├── create_demo_model.py         # Build demo model
│   ├── download_dataset.py          # Dataset download utility
│   ├── prepare_data.py              # Data preprocessing
│   └── train_model.py               # Model training script
├── tests/                           # Unit and integration tests
│   ├── test_classification_router.py
│   ├── test_classification_service.py
│   ├── test_segmentation_router.py
│   └── test_segmentation_service.py
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── README.md
```

### Frontend (`/frontend`)

```
frontend/
├── src/
│   ├── App.jsx                      # Main React component
│   ├── main.jsx                     # React entry point
│   ├── index.css                    # Global styles
│   ├── style.css                    # Additional styles
│   ├── api/
│   │   ├── classificationApi.js     # API calls for classification
│   │   └── segmentationApi.js       # API calls for segmentation
│   ├── components/
│   │   ├── UploadZone.jsx           # File upload component
│   │   ├── ImageCard.jsx            # Image display component
│   │   ├── ClassificationResult.jsx # Classification results display
│   │   └── InfoModal.jsx            # Information modal
│   └── assets/                      # Static assets
├── public/                          # Static resources
├── index.html                       # HTML entry point
├── package.json                     # npm dependencies & scripts
├── tsconfig.json                    # TypeScript configuration
├── vite.config.js                   # Vite build configuration
└── dist/                            # Build output (generated)
```

---

## Core Features

### 1. Image Classification

**Purpose**: Analyze a lesion image and predict the disease category.

**Classification Endpoints**:
- **`POST /api/classify`** - Basic classification
  - Returns: `predicted_label`, `confidence`, `class_index`
  - Accepts: PNG, JPEG, BMP, TIFF images
  - Response Time: ~1-2 seconds

- **`POST /api/classify/gradcam`** - Classification with visual explanation
  - Returns: Same as above + two Grad-CAM visualizations
  - `gradcam_heatmap_image`: Pure heatmap (blue→red importance)
  - `gradcam_overlay_image`: Heatmap blended onto original
  - Useful for explainability and trust

- **`POST /api/classify/stream`** - Test-Time Augmentation (TTA)
  - Streams 100 augmented inference passes via Server-Sent Events
  - Shows real-time progress: epoch count, confidence scores
  - Final result is averaged across all augmentations
  - More robust predictions with confidence tracking

**Classification Models Used**:
1. **Primary**: EfficientNetB0 Keras model (trained on disease dataset)
2. **Fallback**: CLIP-based classifier (zero-shot learning with reference images)
3. **Backup**: Reference classifier (similarity matching)

### 2. Image Segmentation

**Purpose**: Highlight and analyze affected lesion areas within an image.

**Segmentation Endpoint**:
- **`POST /api/segment`** - Fuzzy segmentation + masking
  - Returns multiple processed images:
    - `original_image`: Grayscale version of input
    - `segmented_image`: Fuzzy-membership segmented output
    - `binary_image`: Hybrid adaptive/Otsu threshold mask
    - `gradcam_overlay_image`: 3-band segmentation attention overlay
    - `gradcam_banded_image`: Discrete green/yellow/red heat map
    - `masked_image`: Transparent lesion cutout
  - All images: Base64-encoded PNG strings
  - Metadata: Processing descriptions and timing info

**Segmentation Techniques**:
- **Fuzzy C-Means Clustering**: Determines membership probabilities for image regions
- **Hybrid Thresholding**: Combines adaptive and Otsu's threshold methods
- **Masking**: Creates transparent cutouts of affected areas

### 3. Frontend Features

**Core UI Components**:
- **Upload Zone**: Drag-and-drop interface for image selection
- **Classification Viewer**: Displays predicted label and confidence
- **Segmentation Viewer**: Gallery of segmented images and masks
- **Theme Toggle**: Dark/Light mode switching with localStorage persistence
- **Info Modal**: Displays processing metadata (timing, thresholds, etc.)

**Workflow**:
1. User uploads an image via drag-drop or file browser
2. Frontend makes async API calls to backend
3. Backend processes image through ML pipeline
4. Frontend displays results with interactive visualizations

---

## Data Flow

### Classification Flow

```
User Upload
    ↓
Frontend validates file (type, size)
    ↓
POST /api/classify
    ↓
Backend receives image bytes
    ↓
Classification Service:
  1. Load image from bytes
  2. Preprocess (resize to 224x224, normalize)
  3. Run inference on Keras model
  4. Get class probabilities
  5. Apply confidence threshold
    ↓
Return ClassificationResponse:
{
  predicted_label: "Monkeypox",
  confidence: 0.91,
  class_index: 0,
  message: "Classification completed successfully"
}
    ↓
Frontend displays results
```

### Segmentation Flow

```
User Upload
    ↓
POST /api/segment
    ↓
Backend receives image bytes
    ↓
Segmentation Service:
  1. Load image from bytes
  2. Convert to grayscale
  3. Apply Fuzzy C-Means clustering
  4. Compute binary mask (hybrid threshold)
  5. Create overlays and masked cutout
  6. Encode all outputs to base64 PNG
    ↓
Return SegmentationResponse:
{
  original_image: "base64...",
  segmented_image: "base64...",
  binary_image: "base64...",
  masked_image: "base64...",
  ...metadata
}
    ↓
Frontend displays image gallery
```

---

## Technology Stack

### Backend
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.x | Core language |
| **FastAPI** | Latest | Web framework, API routing |
| **TensorFlow/Keras** | 2.17.0 | Deep learning, model inference |
| **PyTorch** | Latest | Alternative DL framework |
| **Transformers** | Latest | CLIP model for zero-shot classification |
| **scikit-image** | Latest | Image processing & segmentation |
| **scikit-fuzzy** | Latest | Fuzzy logic & clustering |
| **Pillow (PIL)** | Latest | Image I/O and manipulation |
| **NumPy** | Latest | Numerical computing |
| **Matplotlib** | Latest | Visualization (GradCAM heatmaps) |
| **Uvicorn** | Standard | ASGI server for FastAPI |
| **Pydantic** | Latest | Data validation & settings |
| **python-dotenv** | Latest | Environment variable management |

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| **React** | 19.2.4 | UI framework |
| **TypeScript** | 5.9.3 | Static typing (optional) |
| **Vite** | 8.0.1 | Build tool, dev server |
| **JavaScript (ES6+)** | - | Core language |

---

## Configuration

The backend uses environment variables managed via `.env` file:

```env
# Application Settings
APP_NAME = "Monkeypox AI API"
APP_VERSION = "1.1.0"
DEBUG = true

# CORS Settings (allowed frontend origins)
ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]

# Classification Model
CLASSIFICATION_MODEL_PATH = "models/monkeypox_classifier.keras"
CLASSIFICATION_IMAGE_SIZE = 224
CLASSIFICATION_CLASS_NAMES = "Normal,Chickenpox,Measles,Monkeypox"
```

---

## Key Concepts

### Confidence Score
Represents the model's certainty in its prediction (0.0 to 1.0):
- > 0.9: Very confident
- 0.7 - 0.9: Confident
- 0.5 - 0.7: Moderate confidence
- < 0.5: Low confidence / uncertain

### Grad-CAM (Gradient-weighted Class Activation Mapping)
A visualization technique showing:
- Which regions of the image influenced the model's decision
- Heat color mapping: Blue (low importance) → Red (high importance)
- Enhances model interpretability and trustworthiness

### Test-Time Augmentation (TTA)
Processes 100 slightly different versions of the same image:
- Improves prediction robustness
- Reduces noise and variance
- Final prediction is averaged across all augmentations

### Fuzzy Segmentation
Groups similar pixels into clusters based on intensity:
- Non-binary: Pixels have "membership probability" rather than binary membership
- Identifies lesion boundaries with soft edges
- Combined with thresholding for binary mask

---

## Health Check Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | API info and available endpoints |
| `GET /api/classify/health` | Check classifier readiness |
| `GET /api/segment/health` | Check segmentation service status |
| `GET /docs` | Interactive Swagger UI |
| `GET /redoc` | ReDoc documentation |
| `GET /openapi.json` | OpenAPI schema |

---

## Error Handling

The API returns meaningful HTTP status codes and error messages:

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | Success | Classification returned |
| 400 | Bad Request | Empty file, invalid format |
| 415 | Unsupported Media Type | Non-image file uploaded |
| 503 | Service Unavailable | Model not loaded |
| 500 | Internal Server Error | Processing failed |

---

## Performance Considerations

### Classification
- **Latency**: ~1-2 seconds (varies by model size)
- **Memory**: ~500MB for loaded Keras model
- **Throughput**: Can handle multiple concurrent requests

### Segmentation
- **Latency**: ~2-3 seconds (fuzzy clustering is slower)
- **Memory**: ~300MB additional for fuzzy clustering
- **Output Size**: All images base64-encoded (~200-500KB per response)

### Frontend
- **Bundle Size**: ~100-150KB (gzipped)
- **Theme Storage**: localStorage (100 bytes)
- **Request Timeout**: ~30 seconds per upload

---

## Future Enhancement Ideas

- [ ] Batch image processing
- [ ] Model versioning and A/B testing
- [ ] Advanced analytics dashboard
- [ ] Database for storing classification history
- [ ] Mobile app (React Native)
- [ ] Edge deployment (ONNX conversion)
- [ ] Real-time webcam classification
- [ ] Multi-model ensemble voting
- [ ] User authentication and role-based access
- [ ] CloudML integration (Google Cloud, AWS SageMaker)

