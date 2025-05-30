import numpy as np
import pandas as pd
from typing import List
from app.schemas.request_response import LandmarkPoint
import logging


logger = logging.getLogger(__name__)


import numpy as np
import pandas as pd
from typing import List
from app.schemas.request_response import LandmarkPoint
import logging

logger = logging.getLogger(__name__)

def custom_scaling_inference(landmarks: List[LandmarkPoint]) -> np.ndarray:
    """
    Apply the EXACT same custom scaling used during training with robust handling.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
    
    # Convert landmarks to DataFrame format exactly like training
    data = {}
    for i in range(1, 22):  # 1-based indexing like training
        data[f'x{i}'] = landmarks[i-1].x  # Convert to 0-based for landmarks list
        data[f'y{i}'] = landmarks[i-1].y
        data[f'z{i}'] = landmarks[i-1].z
    
    df = pd.DataFrame([data])  # Single row DataFrame
    
    # Apply EXACT same preprocessing as training:
    x_cols = [f'x{i}' for i in range(1, 22)]
    y_cols = [f'y{i}' for i in range(1, 22)]
    
    x_translate = df['x1'].copy()  # WRIST x
    y_translate = df['y1'].copy()  # WRIST y
    x_scale = df['x13'].copy() - x_translate  # MID_FINGER_TIP x relative to WRIST
    y_scale = df['y13'].copy() - y_translate  # MID_FINGER_TIP y relative to WRIST
    
    
    # Use WRIST (x1,y1) as origin
    for col in x_cols:
        df[col] = df[col] - x_translate
    for col in y_cols:
        df[col] = df[col] - y_translate
    
    # Use MID_FINGER_TIP (x13,y13) for scaling
    for col in x_cols:
        df[col] = df[col] / x_scale
    for col in y_cols:
        df[col] = df[col] / y_scale
    
    # Extract the processed values in the EXACT same order as training data
    result_values = []
    for i in range(1, 22):
        result_values.extend([df[f'x{i}'].iloc[0], df[f'y{i}'].iloc[0], df[f'z{i}'].iloc[0]])
    
    result_array = np.array(result_values, dtype=np.float32)
    
    # ROBUST HANDLING: Clip extreme values to reasonable range
    FEATURE_CLIP_RANGE = 20.0
    result_array = np.clip(result_array, -FEATURE_CLIP_RANGE, FEATURE_CLIP_RANGE)
    
    # Check for NaN or inf values
    if np.any(np.isnan(result_array)):
        logger.error("NaN values detected in processed features")
        raise ValueError("NaN values in processed features")
    
    if np.any(np.isinf(result_array)):
        logger.error("Infinite values detected in processed features")
        raise ValueError("Infinite values in processed features")
    
    return result_array

def validate_landmarks(landmarks: List[LandmarkPoint]) -> bool:
    """
    Validate landmark data BEFORE preprocessing.
    Updated for pixel coordinate system used in training.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected exactly 21 landmarks, got {len(landmarks)}")
    
    for i, landmark in enumerate(landmarks):
        if not isinstance(landmark, LandmarkPoint):
            raise ValueError(f"Landmark {i} is not a LandmarkPoint instance")
        
        # Updated ranges for pixel coordinates (based on your training data)
        if not (0.0 <= landmark.x <= 800.0):
            raise ValueError(f"Landmark {i} x-coordinate out of range [0,800]: {landmark.x}")
        
        if not (0.0 <= landmark.y <= 600.0):
            raise ValueError(f"Landmark {i} y-coordinate out of range [0,600]: {landmark.y}")
        
        # Expanded Z range based on your data
        if not (-0.25 <= landmark.z <= 0.25):
            raise ValueError(f"Landmark {i} z-coordinate out of range [-0.25,0.25]: {landmark.z}")
    
    return True

def extract_xyz_coords(landmarks: List[LandmarkPoint]) -> np.ndarray:
    """
    Extract x, y, z coordinates from landmarks for prediction.
    This applies the EXACT same preprocessing as used during training.
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        numpy array of shape (63,) containing processed x, y, z coordinates
    """
    return custom_scaling_inference(landmarks)

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
    Validate landmark data BEFORE preprocessing.
    Based on the actual data distribution from your CSV.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected exactly 21 landmarks, got {len(landmarks)}")
    
    for i, landmark in enumerate(landmarks):
        if not isinstance(landmark, LandmarkPoint):
            raise ValueError(f"Landmark {i} is not a LandmarkPoint instance")
        
        # Expanded ranges based on your actual data
        if not (0.0 <= landmark.x <= 1000.0):
            raise ValueError(f"Landmark {i} x-coordinate out of range [0,1000]: {landmark.x}")
        if not (0.0 <= landmark.y <= 1000.0):
            raise ValueError(f"Landmark {i} y-coordinate out of range [0,1000]: {landmark.y}")
        # FIXED: Expand Z range to handle your actual data
        if not (-0.25 <= landmark.z <= 0.25):  # Expanded from [-0.2, 0.2]
            raise ValueError(f"Landmark {i} z-coordinate out of range [-0.25,0.25]: {landmark.z}")
    
    return True
