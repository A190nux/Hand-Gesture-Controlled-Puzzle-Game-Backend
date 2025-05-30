# Hand Gesture Recognition API

A FastAPI-based REST API for real-time hand gesture recognition using pre-trained XGBoost models and MediaPipe hand landmarks.

## Features

- 🚀 **Fast predictions** using XGBoost
- 📊 **Confidence scores** and probability distributions
- 🔄 **Preprocessing pipeline** matching training methodology
- 🐳 **Docker support** for easy deployment
- 📝 **Comprehensive API documentation** with Swagger UI
- ✅ **Health checks** and monitoring endpoints
- 🧪 **Test suite** included

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd hand-gesture-api
```

### 2. Add Your Model Files

Copy your trained model files to the `models/` directory:
- `*.pkl` or `*.json` - XGBoost model file
- `*label_encoder*.pkl` - Label encoder file

### 3. Run with Docker (Recommended)

```bash
# Build and run
docker-compose up --build

# Or run just the API
docker build -t hand-gesture-api .
docker run -p 8000:8000 hand-gesture-api
```

### 4. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Base URL
- Local: `http://0.0.0.0:8000`
- Docker: `http://0.0.0.0:8000`

### Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Predict Gesture
```bash
POST /predict
Content-Type: application/json

{
  "landmarks": [
    {"x": 0.5, "y": 0.5, "z": 0.0},
    {"x": 0.52, "y": 0.48, "z": 0.01},
    // ... 19 more landmarks (21 total)
  ]
}
```

**Response:**
```json
{
  "gesture": "peace",
  "confidence": 0.95,
  "probabilities": {
    "peace": 0.95,
    "ok": 0.03,
    "stop": 0.02
  },
  "processing_time_ms": 12.5
}
```

#### 3. Get Available Gestures
```bash
GET /gestures
```

#### 4. Model Information
```bash
GET /model/info
```

### Interactive Documentation
Visit `http://0.0.0.0:8000/docs` for Swagger UI documentation.

## Landmark Format

The API expects exactly 21 hand landmarks in MediaPipe format:
- **x, y**: Normalized coordinates in [0, 1]
- **z**: Depth coordinate (can be negative)

Landmarks should be ordered according to MediaPipe hand landmark model:
0. WRIST
1. THUMB_CMC
2. THUMB_MCP
... (see MediaPipe documentation)

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Structure
```
app/
├── main.py              # FastAPI application
├── models/
│   └── predictor.py     # Model loading and prediction
├── schemas/
│   └── request_response.py  # Pydantic models
└── utils/
    └── preprocessing.py # Landmark preprocessing
```

### Adding New Features

1. **New endpoints**: Add to `app/main.py`
2. **New schemas**: Add to `app/schemas/`
3. **New preprocessing**: Add to `app/utils/`
4. **Tests**: Add to `tests/`

## Deployment

### Production Deployment

1. **Environment Variables**:
```bash
export MODEL_PATH=/path/to/model
export ENCODER_PATH=/path/to/encoder
```

2. **Docker Production**:
```bash
docker-compose --profile production up
```

### Performance Tuning

- **Batch predictions**: Modify API to accept multiple landmark sets
- **Model optimization**: Use ONNX or TensorRT for faster inference
- **Caching**: Add Redis for caching frequent predictions
- **Load balancing**: Use multiple API instances behind nginx

## Monitoring

- **Health endpoint**: `/health`
- **Metrics**: Processing time included in responses
- **Logs**: Structured logging with timestamps
- **Docker health checks**: Built-in container health monitoring
- **Average Prediction Time**: A separate metric for prediction time because response time is not a valid indicator for performance in this case
- **Confidence and Landmarks Distribution**: To check for data drift and model degredation
- **CPU and Memory Usage**: For server performance statistics

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Check model files are in `models/` directory
   - Verify file permissions
   - Check logs for specific error messages

2. **Invalid landmarks**:
   - Ensure exactly 21 landmarks
   - Verify coordinate ranges (x,y in [0,1])
   - Check landmark ordering matches MediaPipe

3. **Docker issues**:
   - Ensure port 8000 is available
   - Check Docker daemon is running
   - Verify model files are mounted correctly

### Debug Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## License

[Your License Here]
```

## 13. Setup Commands

Now you can set up everything:

```bash
mkdir hand-gesture-api
cd hand-gesture-api
```

```bash
mkdir -p app/models app/schemas app/utils tests models
```

```bash
touch app/__init__.py app/models/__init__.py app/schemas/__init__.py app/utils/__init__.py tests/__init__.py
```

Then copy your trained model files to the `models/` directory and run:

```bash
docker-compose up --build# Hand-Gesture-Controlled-Puzzle-Game-Backend
