import numpy as np
from typing import List
from app.schemas.request_response import LandmarkPoint

def extract_xy_coords(landmarks: List[LandmarkPoint]) -> np.ndarray:
    """
    Extract x, y coordinates from landmarks for prediction.
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        numpy array of shape (42,) containing x, y coordinates
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
    
    coords = []
    for landmark in landmarks:
        coords.extend([landmark.x, landmark.y])
    
    return np.array(coords, dtype=np.float32)

def normalize_landmarks(landmarks: List[LandmarkPoint]) -> List[LandmarkPoint]:
    """
    Normalize landmarks relative to wrist (landmark 0).
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        Normalized landmarks
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
    
    # Get wrist position (landmark 0)
    wrist = landmarks[0]
    
    # Normalize relative to wrist
    normalized = []
    for landmark in landmarks:
        normalized.append(LandmarkPoint(
            x=landmark.x - wrist.x,
            y=landmark.y - wrist.y,
            z=landmark.z - wrist.z
        ))
    
    return normalized

def validate_landmarks(landmarks: List[LandmarkPoint]) -> bool:
    """
    Validate landmark data.
    
    Args:
        landmarks: List of landmarks to validate
        
    Returns:
        True if valid, raises ValueError if not
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected exactly 21 landmarks, got {len(landmarks)}")
    
    for i, landmark in enumerate(landmarks):
        if not isinstance(landmark, LandmarkPoint):
            raise ValueError(f"Landmark {i} is not a LandmarkPoint instance")
        
        # Check for reasonable coordinate ranges
        if not (-1.0 <= landmark.x <= 1.0):
            raise ValueError(f"Landmark {i} x-coordinate out of range: {landmark.x}")
        if not (-1.0 <= landmark.y <= 1.0):
            raise ValueError(f"Landmark {i} y-coordinate out of range: {landmark.y}")
        if not (-1.0 <= landmark.z <= 1.0):
            raise ValueError(f"Landmark {i} z-coordinate out of range: {landmark.z}")
    
    return True
