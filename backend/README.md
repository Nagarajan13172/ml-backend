# Fuzzy Image Segmentation — FastAPI Backend

A REST API that wraps a fuzzy image segmentation pipeline using **scikit-fuzzy** and **scikit-image**.

## Features
- Upload any image (PNG, JPEG, BMP, TIFF)
- Returns 3 base64-encoded PNG images:
  - **Original** (grayscale)
  - **Fuzzy-segmented** (membership function weighted)
  - **Binary** (Otsu thresholded)
- Swagger UI at `/docs` for instant testing

---

## Project Structure
```
backend/
├── app/
│   ├── main.py                    # FastAPI app + CORS
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── routers/
│   │   └── segmentation.py        # POST /api/segment
│   ├── services/
│   │   └── segmentation_service.py # ML pipeline
│   └── schemas/
│       └── segmentation.py        # Pydantic response model
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Create & activate a virtual environment
```bash
cd ML_DEMO/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment (optional)
```bash
cp .env.example .env
# Edit .env to add your frontend origin to ALLOWED_ORIGINS
```

### 4. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info & links |
| `POST` | `/api/segment` | Upload image → get 3 segmented images |
| `GET` | `/api/segment/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

### Example Request (curl)
```bash
curl -X POST "http://localhost:8000/api/segment" \
  -F "file=@/path/to/your/image.png"
```

### Example Response
```json
{
  "original_image": "<base64 PNG string>",
  "segmented_image": "<base64 PNG string>",
  "binary_image": "<base64 PNG string>",
  "message": "Segmentation completed successfully"
}
```

### Display in Frontend (HTML)
```html
<img src="data:image/png;base64,{{ original_image }}" />
```

---

## CORS Configuration
By default the API allows requests from:
- `http://localhost:3000` (React)
- `http://localhost:5173` (Vite)
- `http://localhost:8080`

Add your frontend URL to `ALLOWED_ORIGINS` in `.env`.
