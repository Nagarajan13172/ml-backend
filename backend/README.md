# Monkeypox AI Backend

A FastAPI backend for two image workflows:
- Classification: upload one lesion image and receive the top predicted label
- Segmentation: upload one lesion image and receive processed diagnostic masks

## Features
- Upload PNG, JPEG, BMP, or TIFF images
- Classification endpoint returns:
  - `predicted_label`
  - `confidence`
- Segmentation endpoint returns:
  - `original_image`
  - `segmented_image`
  - `binary_image`
  - `masked_image`
- Swagger UI at `/docs`

## Project Structure
```text
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── classification.py
│   │   └── segmentation.py
│   ├── schemas/
│   │   ├── classification.py
│   │   └── segmentation.py
│   └── services/
│       ├── classification_service.py
│       └── segmentation_service.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Create and activate a virtual environment
```bash
cd ML_DEMO/backend
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```

Important settings:
- `CLASSIFICATION_MODEL_PATH`: path to your saved `.keras` or `.h5` classifier model
- `CLASSIFICATION_CLASS_NAMES`: comma-separated labels in the same order as model outputs
- `ALLOWED_ORIGINS`: frontend origins allowed by CORS

### 4. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info and links |
| `POST` | `/api/classify` | Upload image and get top predicted label |
| `GET` | `/api/classify/health` | Classifier readiness |
| `POST` | `/api/segment` | Upload image and get segmentation outputs |
| `GET` | `/api/segment/health` | Segmentation health |
| `GET` | `/docs` | Swagger UI |

## Example Classification Request
```bash
curl -X POST "http://localhost:8000/api/classify" \
  -F "file=@/path/to/your/image.png"
```

## Example Classification Response
```json
{
  "predicted_label": "Monkeypox",
  "confidence": 0.91,
  "class_index": 0,
  "message": "Classification completed successfully"
}
```

## Example Segmentation Request
```bash
curl -X POST "http://localhost:8000/api/segment" \
  -F "file=@/path/to/your/image.png"
```

## CORS
By default the API allows:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://localhost:8080`

Update `.env` if your frontend runs elsewhere.
