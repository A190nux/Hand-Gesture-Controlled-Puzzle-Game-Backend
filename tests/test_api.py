import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_valid_landmarks():
    """Test prediction with valid landmarks."""
    # Sample landmarks (21 points with x, y, z coordinates)
    landmarks = []
    for i in range(21):
        landmarks.append({
            "x": 0.5 + (i * 0.01),  # Vary x slightly
            "y": 0.5 + (i * 0.01),  # Vary y slightly  
            "z": 0.0
        })
    
    request_data = {"landmarks": landmarks}
    
    response = client.post("/predict", json=request_data)
    
    # If model is loaded, should return prediction
    if response.status_code == 200:
        data = response.json()
        assert "gesture" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_ms" in data
    else:
        # Model might not be loaded in test environment
        assert response.status_code == 503

def test_predict_invalid_landmarks_count():
    """Test prediction with wrong number of landmarks."""
    # Only 20 landmarks instead of 21
    landmarks = []
    for i in range(20):
        landmarks.append({"x": 0.5, "y": 0.5, "z": 0.0})
    
    request_data = {"landmarks": landmarks}
    
    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_coordinates():
    """Test prediction with invalid coordinate values."""
    landmarks = []
    for i in range(21):
        landmarks.append({
            "x": 2.0,  # Invalid: should be in [0,1]
            "y": 0.5,
            "z": 0.0
        })
    
    request_data = {"landmarks": landmarks}
    
    response = client.post("/predict", json=request_data)
    # Should return 400 for invalid coordinates or 503 if model not loaded
    assert response.status_code in [400, 503]

def test_get_gestures():
    """Test getting available gestures."""
    response = client.get("/gestures")
    
    if response.status_code == 200:
        data = response.json()
        assert "gestures" in data
        assert "count" in data
        assert isinstance(data["gestures"], list)
    else:
        # Model might not be loaded
        assert response.status_code == 503

def test_model_info():
    """Test getting model information."""
    response = client.get("/model/info")
    
    if response.status_code == 200:
        data = response.json()
        assert "loaded" in data
    else:
        # Model might not be loaded
        assert response.status_code == 503