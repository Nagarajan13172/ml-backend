# Monkeypox AI Project - Setup & Run Guide

This guide provides step-by-step instructions to set up and run both the backend and frontend components of the Monkeypox AI project.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Running the Application](#running-the-application)
5. [Verification & Testing](#verification--testing)
6. [Troubleshooting](#troubleshooting)
7. [Additional Commands](#additional-commands)

---

## System Requirements

### Operating System
- **Windows** 10/11
- **macOS** 10.14+
- **Linux** (Ubuntu 18.04+)

### Software
- **Python** 3.8 or higher
- **Node.js** 16+ with npm
- **Git** (for cloning repository)
- **RAM**: Minimum 8GB (16GB recommended for ML models)
- **Storage**: At least 10GB free space (for models and datasets)

### Verification

Check installed versions:

```bash
# Check Python version
python --version

# Check Node.js and npm version
node --version
npm --version
```

---

## Backend Setup

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Create Virtual Environment

Creates an isolated Python environment for project dependencies.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

✓ Verify activation: Your terminal should show `(venv)` prefix.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs:
- **FastAPI** - Web framework
- **TensorFlow** - Deep learning framework
- **PyTorch** - Alternative DL framework
- **scikit-image** - Image processing
- **Pillow** - Image I/O
- Other utilities (numpy, matplotlib, pydantic, etc.)

**Expected time**: 5-10 minutes (depends on internet speed and system)

⚠️ **Note**: On Apple Silicon Macs, TensorFlow-macos will be installed instead.

### Step 4: Configure Environment Variables

Copy the example environment file:

```bash
copy .env.example .env
```

Edit `.env` file with your settings:

```env
# Application Settings
APP_NAME = "Monkeypox AI API"
APP_VERSION = "1.1.0"
DEBUG = true

# CORS Settings (allowed frontend origins)
ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]

# Classification Model Path (relative to backend folder)
CLASSIFICATION_MODEL_PATH = "models/monkeypox_classifier.keras"

# Model Configuration
CLASSIFICATION_IMAGE_SIZE = 224
CLASSIFICATION_CLASS_NAMES = "Normal,Chickenpox,Measles,Monkeypox"
```

**Key Configuration Details**:

| Setting | Description | Example |
|---------|-------------|---------|
| `CLASSIFICATION_MODEL_PATH` | Path to trained Keras model | `models/monkeypox_classifier.keras` |
| `CLASSIFICATION_CLASS_NAMES` | Comma-separated class labels (order must match model output) | `Normal,Chickenpox,Measles,Monkeypox` |
| `ALLOWED_ORIGINS` | Frontend URLs allowed by CORS | Include all frontend dev/prod URLs |
| `DEBUG` | Development/Debug mode | `true` for development, `false` for production |

### Step 5: Verify Model Files

Ensure trained model exists in `backend/models/`:

```bash
# List model files
dir models/
```

Expected files:
- `monkeypox_classifier.keras` - Primary model (required)
- `monkeypox_classifier_ft.keras` - Fine-tuned variant (optional)
- `README.md` - Model documentation

If models are missing, see [Additional Commands](#additional-commands) for training/download options.

### Backend Verification

Test the backend is ready:

```bash
# Check if FastAPI app can be imported
python -c "from app.main import app; print('✓ FastAPI app imported successfully')"

# Check if TensorFlow is available
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} loaded')"
```

---

## Frontend Setup

### Step 1: Navigate to Frontend Directory

From project root:

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

This installs:
- **React** - UI framework
- **Vite** - Build tool
- **TypeScript** - Optional static typing
- Development dependencies

**Expected time**: 2-5 minutes

### Step 3: Verify Setup

```bash
npm --version
node --version
```

### Frontend Verification

Check Vite configuration:

```bash
# Verify vite config exists
dir vite.config.js

# List frontend structure
dir /s src
```

---

## Running the Application

### Option 1: Run Both Services (Recommended for Development)

Open **two separate terminals**:

#### Terminal 1 - Backend Service

```bash
cd backend
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

uvicorn app.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
```

✓ Backend is ready when you see `Uvicorn running on http://127.0.0.1:8000`

#### Terminal 2 - Frontend Service

```bash
cd frontend
npm run dev
```

Expected output:
```
  VITE v8.0.1  ready in 123 ms

  ➜  Local:   http://127.0.0.1:5173/
  ➜  press h to show help
```

✓ Frontend is ready when you see the local URL

### Option 2: Using VS Code Terminal Manager

If using VS Code:

1. Open **Terminal** → **New Terminal** (Ctrl+`)
2. In first terminal: Run backend command
3. In same terminal panel: Create new terminal (splits window or tab)
4. In second terminal: Run frontend command
5. Both services run simultaneously in split view

### Option 3: Run in Production Mode

Build frontend for production and serve both from backend:

```bash
# Build frontend
cd frontend
npm run build

# Backend still runs normally
cd ../backend
uvicorn app.main:app --port 8000
```

Then access at `http://localhost:8000` (backend serves built frontend)

---

## Verification & Testing

### Health Checks

Once both services are running:

#### 1. Backend Health

Open browser or make request:

```bash
# Check API is running
curl http://localhost:8000/

# Check classification health
curl http://localhost:8000/api/classify/health

# Check segmentation health
curl http://localhost:8000/api/segment/health

# View API documentation
# Go to http://localhost:8000/docs in browser
```

Expected response from `/`:
```json
{
  "message": "Welcome to Monkeypox AI API v1.1.0",
  "docs": "/docs",
  "classification_health": "/api/classify/health",
  "health": "/api/segment/health"
}
```

#### 2. Frontend Access

Open `http://127.0.0.1:5173/` in browser. You should see:
- Upload zone for image upload
- Tabs for Classification and Segmentation
- Theme toggle button
- Loading states when processing

#### 3. Test Classification

**Via Browser UI**:
1. Go to `http://127.0.0.1:5173/`
2. Click upload zone or drag-drop an image
3. Click "Classify" button
4. See result: predicted label and confidence

**Via cURL** (Terminal):
```bash
curl -X POST "http://localhost:8000/api/classify" \
  -F "file=@path/to/image.png"
```

Expected response:
```json
{
  "predicted_label": "Normal",
  "confidence": 0.85,
  "class_index": 0,
  "message": "Classification completed successfully"
}
```

#### 4. Test Segmentation

**Via Browser UI**:
1. Go to Classification tab
2. Upload image and click "Segment"
3. View segmented images gallery

**Via cURL**:
```bash
curl -X POST "http://localhost:8000/api/segment" \
  -F "file=@path/to/image.png" \
  -o response.json
```

### Run Tests

Execute test suite:

```bash
# From backend directory
cd backend

# Run all tests
pytest

# Run specific test file
pytest tests/test_classification_service.py -v

# Run with coverage report
pytest --cov=app tests/
```

---

## Troubleshooting

### Backend Issues

#### Issue: "ModuleNotFoundError: No module named 'app'"

**Cause**: Not in correct directory or venv not activated

**Solution**:
```bash
# Ensure you're in backend directory
pwd  # or cd backend

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Verify
python -c "import app"
```

#### Issue: "Port 8000 already in use"

**Cause**: Another process using port 8000

**Solution**:
```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000

# macOS/Linux:
lsof -i :8000

# Kill process (replace PID with actual process ID)
# Windows:
taskkill /PID 12345 /F

# macOS/Linux:
kill -9 12345

# Or use different port:
uvicorn app.main:app --reload --port 8001
```

#### Issue: TensorFlow import fails / CUDA errors

**Cause**: Model or TensorFlow library issues

**Solution**:
```bash
# Reinstall TensorFlow
pip install --upgrade tensorflow --force-reinstall

# Verify installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

#### Issue: ".keras model file not found"

**Cause**: Model path incorrect or file missing

**Solution**:
```bash
# Check if model exists
ls -la backend/models/

# If missing, download or train model (see Additional Commands)

# Update CLASSIFICATION_MODEL_PATH in .env if path is wrong
```

### Frontend Issues

#### Issue: "npm: command not found"

**Cause**: Node.js not installed or not in PATH

**Solution**:
```bash
# Install Node.js from https://nodejs.org/

# Verify installation
node --version
npm --version
```

#### Issue: "Port 5173 already in use"

**Cause**: Another Vite dev server running

**Solution**:
```bash
# Find and kill process (similar to backend)
# Or let Vite auto-select next available port

# Or specify different port:
npm run dev -- --port 5174
```

#### Issue: Blank page or "Failed to connect to API"

**Cause**: Backend not running or CORS misconfigured

**Solution**:
```bash
# 1. Verify backend is running at http://localhost:8000

# 2. Check ALLOWED_ORIGINS in backend/.env includes frontend URL:
ALLOWED_ORIGINS = ["http://localhost:5173"]

# 3. Restart backend after .env changes

# 4. Clear browser cache (Ctrl+Shift+Delete)
```

#### Issue: "React version mismatch" or Module errors

**Cause**: Package corruption or npm lock conflicts

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install

# Restart dev server
npm run dev
```

### CORS (Cross-Origin) Issues

If frontend can't reach backend:

1. **Check backend is running**: `http://localhost:8000`
2. **Verify ALLOWED_ORIGINS** in `backend/.env`:
   ```env
   ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
   ```
3. **Restart backend** after changing `.env`
4. **Check browser console** (F12) for CORS error details

---

## Additional Commands

### Backend Commands

#### Train a New Model

```bash
cd backend
python scripts/train_model.py
```

#### Prepare Dataset

```bash
python scripts/prepare_data.py
```

#### Create Demo Model (for testing)

```bash
python scripts/create_demo_model.py
```

#### Build CLIP Reference Database

```bash
python scripts/build_reference_db.py
```

### Frontend Commands

#### Build for Production

```bash
cd frontend
npm run build
```

Output will be in `dist/` folder (ready to deploy)

#### Preview Production Build

```bash
npm run preview
```

#### Format/Lint Code

```bash
# (if ESLint/Prettier configured)
npm run lint
npm run format
```

### Development Server Advanced Options

#### Backend - Specific reload behavior

```bash
# No auto-reload (production-like)
uvicorn app.main:app --port 8000

# With custom log level
uvicorn app.main:app --reload --log-level debug

# With access logs
uvicorn app.main:app --reload --access-log
```

#### Frontend - Debug mode

```bash
# Verbose output
npm run dev -- --debug

# Different host
npm run dev -- --host 0.0.0.0

# Different port
npm run dev -- --port 5174
```

### Database/Data Commands

#### Download Full Dataset

```bash
cd backend
python scripts/download_dataset.py
```

#### Check Dataset Structure

```bash
# List training data statistics
python -c "
import os
from pathlib import Path

data_dir = Path('data/train')
for class_dir in data_dir.iterdir():
    count = len(list(class_dir.glob('*')))
    print(f'{class_dir.name}: {count} images')
"
```

---

## Common Workflows

### First-Time Setup (Complete)

```bash
# 1. Backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env
# Edit .env if needed

# 2. Frontend
cd ../frontend
npm install

# 3. Start services (in separate terminals)
# Terminal 1:
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# Terminal 2:
cd frontend
npm run dev

# 4. Access at http://127.0.0.1:5173/
```

### Daily Development

```bash
# Terminal 1 - Backend (if not running)
cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend (if not running)
cd frontend && npm run dev

# Edit code in files - both will auto-reload
```

### Prepare for Deployment

```bash
# Backend - disable debug mode
# Edit backend/.env:
# DEBUG = false

# Frontend - build production version
cd frontend && npm run build

# Result: frontend/dist/ contains optimized assets
```

### Clean & Fresh Start

```bash
# Backend
cd backend
deactivate  # Exit venv
rm -rf venv  # Remove environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Frontend
cd ../frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## Port Reference

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| Backend (Uvicorn) | 8000 | `http://localhost:8000/` | API server |
| Backend (Swagger) | 8000 | `http://localhost:8000/docs` | API documentation |
| Frontend (Vite Dev) | 5173 | `http://127.0.0.1:5173/` | Development server |
| Alternative Backend | 8001 | `http://localhost:8001/` | If 8000 is busy |
| Alternative Frontend | 5174 | `http://127.0.0.1:5174/` | If 5173 is busy |

---

## Performance Tips

### Backend Optimization

1. **Disable reload in production**: Remove `--reload` flag
2. **Use workers**: `uvicorn app.main:app --port 8000 --workers 4`
3. **GPU acceleration**: Ensure TensorFlow detects GPU
4. **Cache models**: Pre-load models on startup

### Frontend Optimization

1. **Build for production**: `npm run build`
2. **Enable gzip compression**: Configure web server
3. **CDN caching**: Cache static assets
4. **Lazy loading**: Load components on demand

---

## Getting Help

1. **Check Logs**: Look at terminal output from backend/frontend
2. **API Docs**: Visit `http://localhost:8000/docs` for interactive API
3. **Error Messages**: Read error details carefully
4. **Browser Console**: F12 → Console for frontend errors
5. **Test Files**: Review `backend/tests/` for usage examples

### Docker Deployment (Optional)

#### Backend Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile
```dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY frontend/ .
RUN npm install
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Running with Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - CLASSIFICATION_MODEL_PATH=/app/models/monkeypox_classifier.keras
      - CLASSIFICATION_CLASS_NAMES=Normal,Chickenpox,Measles,Monkeypox
      - ALLOWED_ORIGINS=http://localhost:3000,http://frontend:80
    volumes:
      - ./backend/models:/app/models

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

---

## Performance Optimization Tips

### Backend Optimization
1. **Model Quantization**: Convert `.keras` to TensorFlow Lite for faster inference
2. **Caching**: Cache model in memory (already done by default)
3. **Batch Processing**: Modify endpoints to accept multiple images
4. **GPU Support**: Install CUDA and cuDNN for faster inference

### Frontend Optimization
1. **Code Splitting**: Lazy load components
2. **Image Compression**: Compress uploaded images before sending
3. **Caching**: Store results locally
4. **CDN**: Serve frontend from CDN in production

---

## Monitoring & Logging

### Backend Logs
```bash
# View logs with timestamps
uvicorn app.main:app --reload --log-level debug
```

Log levels: `critical`, `error`, `warning`, `info`, `debug`

### Frontend Debugging
1. Open DevTools: F12
2. Console tab: See JavaScript errors
3. Network tab: Monitor API calls
4. Application tab: Check localStorage and cookies

---

## Quick Reference Commands

```bash
# Backend Activation (Windows)
backend\venv\Scripts\activate

# Backend Activation (macOS/Linux)
source backend/venv/bin/activate

# Start Backend Server
uvicorn app.main:app --reload --port 8000

# Start Frontend Dev Server
npm run dev

# Build Frontend for Production
npm run build

# Run Backend Tests
pytest tests/ -v

# Install Additional Backend Dependencies
pip install <package-name>

# Freeze Backend Dependencies
pip freeze > requirements.txt

# Clear Frontend Cache
npm cache clean --force

# View API Documentation
# Open browser to: http://localhost:8000/docs
```

---

## Next Steps

1. ✅ Complete setup above
2. 🧪 Test with sample images from `backend/data/` directories
3. 📚 Review API documentation at `/docs`
4. 🔧 Customize model and class labels as needed
5. 🚀 Deploy using Docker or cloud platform
6. 📖 Refer to `PROJECT_OVERVIEW.md` for architecture details

---

## Getting Help

- **API Issues**: Check `http://localhost:8000/docs` for endpoint details
- **Model Issues**: See `backend/models/README.md`
- **Data Issues**: See `backend/scripts/` for data preparation
- **Frontend Issues**: Check browser console (F12) for errors
- **Dependencies**: Review `backend/requirements.txt` and `frontend/package.json`

