# Monkeypox AI - Project Overview

## Table of Contents
1. [Project Summary](#project-summary)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Backend Structure](#backend-structure)
5. [Frontend Structure](#frontend-structure)
6. [Workflows](#workflows)
7. [Key Features](#key-features)
8. [Data Flow](#data-flow)

---

## Project Summary

**Monkeypox AI** is a full-stack machine learning application designed to assist in medical image diagnosis of infectious diseases. It provides two primary capabilities:

- **Classification**: Classifies lesion images into disease categories (Monkeypox, Chickenpox, Measles, Normal)
- **Segmentation**: Generates diagnostic visual masks to identify and highlight affected areas in lesion images

The project combines a **FastAPI backend** with a **React frontend** to create an intuitive, web-based diagnostic tool.

### Use Cases
- Quick classification of skin lesion images for differential diagnosis support
- Visual segmentation to identify lesion boundaries and affected areas
- Real-time Grad-CAM visualizations to understand model decision-making
- Deployment-ready API endpoints for integration with medical systems

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────┐
│              React Frontend (Port 5173)              │
│  - Image Upload Interface                           │
│  - Classification Results Display                   │
│  - Segmentation Visualizations                      │
│  - Grad-CAM Heatmaps                                │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ HTTP/REST API
                   │
┌──────────────────▼──────────────────────────────────┐
│         FastAPI Backend (Port 8000)                  │
│  ┌────────────────────────────────────────────────┐ │
│  │  API Layer (Routers)                           │ │
│  │  - /api/classify           (POST)              │ │
│  │  - /api/classify/gradcam   (POST)              │ │
│  │  - /api/segment            (POST)              │ │
│  │  - /api/segment/health     (GET)               │ │
│  └────────────────────────────────────────────────┘ │
│                   │                                  │
│  ┌────────────────▼────────────────────────────────┐ │
│  │  Service Layer                                  │ │
│  │  - ClassificationService                       │ │
│  │  - SegmentationService                         │ │
│  │  - CLIPClassifier                              │ │
│  │  - ReferenceClassifier                         │ │
│  └────────────────┬────────────────────────────────┘ │
│                   │                                  │
│  ┌────────────────▼────────────────────────────────┐ │
│  │  ML Models & Processing                        │ │
│  │  - TensorFlow/Keras Models                     │ │
│  │  - Fuzzy Logic Segmentation                    │ │
│  │  - Image Preprocessing (PIL, scikit-image)    │ │
│  └────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Persistent Storage                          │
│  - /models/                  (Trained ML models)    │
│  - /data/                    (Training/Val data)    │
│  - .env                      (Configuration)        │
└───────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI | REST API server |
| Web Server | Uvicorn | ASGI application server |
| ML Framework | TensorFlow 2.17 | Deep learning models |
| Image Processing | PIL, scikit-image | Image manipulation & segmentation |
| API Validation | Pydantic | Request/response validation |
| Configuration | python-dotenv | Environment management |
| CORS Handling | FastAPI Middleware | Cross-origin requests |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 19.2 | UI component library |
| Build Tool | Vite 8.0 | Fast bundler & dev server |
| Language | TypeScript/JSX | Type-safe development |
| Styling | CSS3 | Component styling |

### ML Libraries
- **TensorFlow**: Neural network models (EfficientNetB0-based classifiers)
- **scikit-fuzzy**: Fuzzy logic for image segmentation
- **scikit-image**: Advanced image processing algorithms
- **Transformers/Torch**: Support for CLIP and other transformer models
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization utilities

---

## Backend Structure

### Directory Layout

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization
│   ├── config.py               # Settings & configuration
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── classification.py   # Classification endpoints
│   │   └── segmentation.py     # Segmentation endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── classification.py   # Pydantic models for classification
│   │   └── segmentation.py     # Pydantic models for segmentation
│   └── services/
│       ├── __init__.py
│       ├── classification_service.py    # Classification logic
│       ├── clip_classifier.py           # CLIP model classifier
│       ├── reference_classifier.py      # Reference-based classifier
│       └── segmentation_service.py      # Segmentation logic
├── tests/                      # Unit tests
├── models/                     # Pre-trained ML models (.keras/.h5)
├── scripts/                    # Utility scripts
├── data/                       # Training/validation datasets
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md
```

### Key Files Explained

#### `app/main.py`
- Initializes FastAPI application
- Sets up CORS middleware to allow frontend requests
- Includes routers for classification and segmentation
- Provides root endpoint with links to available services

#### `app/config.py`
- Loads environment variables from `.env` file
- Defines configuration settings like:
  - `CLASSIFICATION_MODEL_PATH`: Path to trained classifier
  - `CLASSIFICATION_CLASS_NAMES`: Disease labels
  - `ALLOWED_ORIGINS`: Frontend domains allowed to access API
  - `CLASSIFICATION_IMAGE_SIZE`: Input image dimensions (224x224)

#### `app/routers/classification.py`
**Endpoints:**
- `POST /api/classify` - Classify single image, return top prediction
- `POST /api/classify/gradcam` - Classify with Grad-CAM heatmap visualization
- `GET /api/classify/health` - Health check

#### `app/routers/segmentation.py`
**Endpoints:**
- `POST /api/segment` - Process image with fuzzy segmentation
- `GET /api/segment/health` - Health check

#### `app/services/classification_service.py`
- Loads trained TensorFlow/Keras model
- Preprocesses uploaded images (resize, normalize)
- Performs inference
- Returns predictions with confidence scores

#### `app/services/segmentation_service.py`
- Implements fuzzy membership-based image segmentation
- Applies adaptive/Otsu thresholding for binary masks
- Generates masked cutouts
- Returns multiple processed image outputs (base64-encoded)

#### `app/schemas/*.py`
- Pydantic models defining request/response structures
- Example: `ClassificationResponse` with fields for label, confidence, class_index
- Automatic validation and documentation

---

## Frontend Structure

```
frontend/
├── src/
│   ├── main.jsx                # React app entry point
│   ├── App.jsx                 # Main app component
│   ├── index.css               # Global styles
│   ├── api/
│   │   ├── classificationApi.js    # Classification API client
│   │   └── segmentationApi.js      # Segmentation API client
│   └── components/
│       ├── UploadZone.jsx      # Drag-drop file upload
│       ├── ImageCard.jsx       # Image result display
│       └── InfoModal.jsx       # Details modal
├── package.json                # Dependencies & scripts
├── tsconfig.json               # TypeScript config
├── vite.config.js              # Vite build config
├── index.html                  # HTML entry point
└── public/                     # Static assets
```

### Key Components

#### `App.jsx`
Main application container managing:
- File upload state
- Classification results (with/without Grad-CAM)
- Segmentation results
- Error handling
- Theme management (light/dark mode)

#### `UploadZone.jsx`
- Drag-and-drop file upload interface
- File type validation (PNG, JPEG, BMP, TIFF)
- Visual feedback during upload

#### `ImageCard.jsx`
- Displays uploaded images
- Shows classification results with confidence scores
- Renders segmentation masks and overlays
- Displays Grad-CAM heatmaps

#### `classificationApi.js`
API client functions:
- `classifyImage(file)` - POST request to `/api/classify`
- `classifyWithGradcam(file)` - POST request to `/api/classify/gradcam`
- Handles multipart form data
- Manages error responses

#### `segmentationApi.js`
API client functions:
- `segmentImage(file)` - POST request to `/api/segment`
- Returns multiple processed image outputs

---

## Workflows

### 1. Classification Workflow

```
User Uploads Image
        ↓
Frontend validates file type
        ↓
POST /api/classify with multipart form-data
        ↓
Backend receives image bytes
        ↓
ImagePreprocessor resizes & normalizes (224x224)
        ↓
TensorFlow Model inference
        ↓
Post-process predictions
        ↓
Return: {predicted_label, confidence, class_index, message}
        ↓
Frontend displays result with confidence bar
```

**Response Example:**
```json
{
  "predicted_label": "Monkeypox",
  "confidence": 0.92,
  "class_index": 2,
  "message": "Classification completed successfully"
}
```

### 2. Classification with Grad-CAM

```
User Uploads Image + Requests Grad-CAM
        ↓
POST /api/classify/gradcam
        ↓
Perform classification (as above)
        ↓
Compute Grad-CAM heatmap
        ↓
Generate two visualizations:
  - Standalone heatmap (blue→red intensity)
  - Heatmap overlay on original image
        ↓
Encode images as base64 PNG
        ↓
Return: {predicted_label, confidence, gradcam_heatmap_image, gradcam_overlay_image}
        ↓
Frontend displays heatmap showing model attention regions
```

### 3. Segmentation Workflow

```
User Uploads Image
        ↓
POST /api/segment
        ↓
Convert to grayscale
        ↓
Apply fuzzy logic segmentation
        ↓
Generate binary mask (adaptive + Otsu thresholding)
        ↓
Create three-band segmentation overlay
        ↓
Generate banded categorical intensity map
        ↓
Apply mask to create transparent lesion cutout
        ↓
Encode all results as base64 PNG
        ↓
Return: {
  original_image,
  segmented_image,
  binary_image,
  gradcam_overlay_image,
  gradcam_banded_image,
  masked_image,
  binary_details,
  gradcam_details
}
        ↓
Frontend displays gallery of processed images
```

---

## Key Features

### 1. **Multi-Label Classification**
- Supports configurable disease labels (Monkeypox, Chickenpox, Measles, Normal)
- Returns top prediction with confidence score
- Fast inference (~100-200ms on CPU)

### 2. **Explainable AI with Grad-CAM**
- Visualizes which regions the model focused on for classification
- Generates heatmaps to show model reasoning
- Supports both standalone and overlay visualizations

### 3. **Advanced Image Segmentation**
- Fuzzy membership-based segmentation for soft boundaries
- Hybrid thresholding (adaptive + Otsu) for robust binary masks
- Categorical intensity mapping for multi-level segmentation

### 4. **Multiple Classifier Support**
- Primary: Trained Keras/TensorFlow model
- Fallback: CLIP-based zero-shot classifier
- Fallback: Reference image database classifier

### 5. **RESTful API Design**
- OpenAPI/Swagger documentation at `/docs`
- Health check endpoints for monitoring
- CORS enabled for cross-origin requests
- Comprehensive error handling with descriptive messages

### 6. **Flexible Image Input**
- Supports PNG, JPEG, BMP, TIFF formats
- Automatic grayscale conversion for segmentation
- Robust preprocessing and validation

### 7. **Base64 Image Encoding**
- All processed images returned as base64-encoded PNG strings
- Eliminates need for separate file downloads
- Direct integration with image elements in frontend

---

## Data Flow

### Complete End-to-End Example

#### Classification Flow:
```
1. User selects image.jpg on frontend
2. Frontend validates content-type
3. Frontend: POST /api/classify with FormData
   - Body: multipart/form-data with "file" field
4. Backend receives bytes from UploadFile
5. classification_service.classify_image() called:
   a. Load model from CLASSIFICATION_MODEL_PATH
   b. Resize image to (224, 224)
   c. Normalize pixel values (0-1)
   d. Run model.predict()
   e. Get class probabilities
   f. Find argmax (top class)
   g. Look up class name from CLASSIFICATION_CLASS_NAMES
   h. Calculate confidence
6. Return ClassificationResponse
7. Frontend receives JSON & displays result
```

#### Segmentation Flow:
```
1. User selects image.jpg on frontend
2. Frontend: POST /api/segment with FormData
3. Backend receives bytes
4. segmentation_service.process_image() called:
   a. Load image with PIL
   b. Convert to RGB then grayscale
   c. Apply fuzzy triangular membership function
   d. Compute adaptive binary threshold
   e. Compute Otsu threshold
   f. Combine thresholds
   g. Generate masked cutout
   h. Create three-band overlay (G/Y/R channels)
   i. Encode all outputs as base64 PNG
5. Return SegmentationResponse with 8 image fields
6. Frontend receives JSON & renders image gallery
```

---

## Configuration Management

### Environment Variables (.env)

```bash
# API Configuration
APP_NAME=Monkeypox AI API
APP_VERSION=1.1.0
DEBUG=true

# Model Configuration
CLASSIFICATION_MODEL_PATH=models/monkeypox_classifier.keras
CLASSIFICATION_CLASS_NAMES=Normal,Chickenpox,Measles,Monkeypox

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080
```

- `CLASSIFICATION_MODEL_PATH`: Absolute or relative path to `.keras` model file
- `CLASSIFICATION_CLASS_NAMES`: Comma-separated labels (order must match model outputs)
- `ALLOWED_ORIGINS`: Comma-separated list of frontend domains

---

## Error Handling

### Backend Error Responses

| Status Code | Scenario | Example |
|------------|----------|---------|
| 400 | Invalid input (empty file, wrong format) | `"Uploaded file is empty."` |
| 415 | Unsupported media type | `"Unsupported file type 'image/webp'. Allowed types: image/png, image/jpeg..."` |
| 503 | Model not ready/not loaded | `"Classifier not ready. Check MODEL_PATH and dependencies."` |
| 500 | Processing error | `"Image classification failed: ..."` |

### Frontend Error Handling
- Displays user-friendly error messages
- Provides retry capability
- Logs detailed errors to console
- Shows loading states during processing

---

## Performance Considerations

### Backend
- **Image Preprocessing**: ~10-20ms
- **Model Inference**: ~50-150ms (depends on model size and hardware)
- **Segmentation Processing**: ~30-80ms
- **Grad-CAM Generation**: ~100-200ms
- **Total Response Time**: 200-500ms typical

### Frontend
- Asynchronous API calls (non-blocking UI)
- Lazy loading of images
- Theme persistence in localStorage
- Responsive design for mobile/tablet

---

## Security Considerations

1. **File Validation**: Content-type checking on both frontend and backend
2. **File Size Limits**: Can be configured in FastAPI settings
3. **CORS Configuration**: Restrict to known frontend origins
4. **Error Messages**: Informative but don't expose system internals
5. **Environment Secrets**: Sensitive paths/credentials in `.env`

---

## Extensibility

The architecture supports:
- Adding new disease classifiers
- Integrating different ML models
- Adding new segmentation algorithms
- Supporting additional image formats
- Implementing user authentication
- Adding data logging/analytics

---

## Development Notes

- Models are TensorFlow/Keras format (`.keras` files)
- Image preprocessing follows EfficientNetB0 standards
- Segmentation uses scikit-fuzzy for membership functions
- All responses are either JSON or base64-encoded images
- No database required (stateless API)
- Can be containerized with Docker for deployment

---

## Related Documentation
- See `SETUP_AND_RUN.md` for installation and execution instructions
- Refer to backend `README.md` for API endpoint details
- Check model training scripts in `backend/scripts/` for data preparation
