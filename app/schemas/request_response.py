from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Optional

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float

class HandLandmarksRequest(BaseModel):
    landmarks: List[LandmarkPoint]
    
    @field_validator('landmarks')
    @classmethod
    def validate_landmarks_count(cls, v):
        if len(v) != 21:
            raise ValueError('Must provide exactly 21 landmarks')
        return v

class PredictionResponse(BaseModel):
    gesture: str
    confidence: float
    probabilities: dict
    processing_time_ms: float

# Fix the timestamp issue
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    timestamp: Optional[str] = None  # Make it optional

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""

class GesturesResponse(BaseModel):
    gestures: List[str]
    count: int

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    loaded: bool
    model_type: str = ""
    classes: List[str] = []
    features: int = 0
