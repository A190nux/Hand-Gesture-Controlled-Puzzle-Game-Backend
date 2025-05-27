from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager

from app.models.predictor import GesturePredictor
from app.schemas.request_response import (
    HandLandmarksRequest, 
    PredictionResponse, 
    HealthResponse,
    ErrorResponse
)
from app.utils.preprocessing import validate_landmarks
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    global predictor
    try:
        predictor = GesturePredictor()
        predictor.load_model()
        logger.info("✅ Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")
        # You might want to exit here or continue without model
        
    yield
    
    # Shutdown
    logger.info("🛑 Application shutting down")

# Create FastAPI app
app = FastAPI(
    title="Hand Gesture Recognition API",
    description="API for predicting hand gestures from MediaPipe landmarks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hand Gesture Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and getattr(predictor, 'is_loaded', False),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(request: HandLandmarksRequest):
    """
    Predict hand gesture from landmarks.
    
    Args:
        request: HandLandmarksRequest containing 21 hand landmarks
        
    Returns:
        PredictionResponse with gesture prediction and confidence
    """
    global predictor
    
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Validate landmarks
    if not validate_landmarks(request.landmarks):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid landmark coordinates. X and Y should be in [0,1], Z should be reasonable."
        )
    
    try:
        start_time = time.time()
        
        # Make prediction
        gesture, confidence, probabilities = predictor.predict(request.landmarks)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return PredictionResponse(
            gesture=gesture,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return predictor.get_model_info()

@app.get("/gestures")
async def get_available_gestures():
    """Get list of available gesture classes."""
    global predictor
    
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "gestures": predictor.class_names,
        "count": len(predictor.class_names)
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)