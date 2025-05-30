import pickle
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import json
from app.utils.preprocessing import extract_xyz_coords  # CHANGED: Use XYZ instead of XY
from app.schemas.request_response import LandmarkPoint

logger = logging.getLogger(__name__)

class GesturePredictor:
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the gesture predictor.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.label_encoder = None
        self.class_names = None
        self.model_info = None
        self.is_loaded = False
        
    def load_model(self, 
                   model_path: str = None,
                   encoder_path: str = None,
                   class_names_path: str = None):
        """
        Load the trained model and preprocessing components.
        """
        try:
            logger.info(f"Looking for model files in: {self.model_dir}")
            
            # List all files in models directory
            all_files = list(self.model_dir.glob("*"))
            logger.info(f"Found files: {[f.name for f in all_files]}")
            
            # Find the latest model files (by timestamp)
            if model_path is None:
                model_files = list(self.model_dir.glob("xgboost_model_*.pkl"))
                if not model_files:
                    raise FileNotFoundError(f"No XGBoost model file found in {self.model_dir}")
                # Get the latest model file
                model_path = sorted(model_files)[-1]
                logger.info(f"Using model file: {model_path}")
            
            # Find corresponding encoder file
            if encoder_path is None:
                # Extract timestamp from model filename
                model_timestamp = model_path.name.split('_')[-1].replace('.pkl', '')
                encoder_files = list(self.model_dir.glob(f"label_encoder_{model_timestamp}.pkl"))
                if not encoder_files:
                    # Fall back to any encoder file
                    encoder_files = list(self.model_dir.glob("label_encoder_*.pkl"))
                if not encoder_files:
                    raise FileNotFoundError(f"No label encoder file found in {self.model_dir}")
                encoder_path = encoder_files[0]
                logger.info(f"Using encoder file: {encoder_path}")
            
            # Find class names file
            if class_names_path is None:
                model_timestamp = model_path.name.split('_')[-1].replace('.pkl', '')
                class_files = list(self.model_dir.glob(f"class_names_{model_timestamp}.json"))
                if not class_files:
                    class_files = list(self.model_dir.glob("class_names_*.json"))
                if class_files:
                    class_names_path = class_files[0]
                    logger.info(f"Using class names file: {class_names_path}")
            
            # Load model using joblib (safer for your pickle files)
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully using joblib")
            
            # Load label encoder
            logger.info(f"Loading encoder from: {encoder_path}")
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("✅ Label encoder loaded successfully")
            
            # Load class names
            if class_names_path and class_names_path.exists():
                with open(class_names_path, 'r') as f:
                    class_data = json.load(f)
                    if isinstance(class_data, list):
                        self.class_names = class_data
                    elif isinstance(class_data, dict) and 'classes' in class_data:
                        self.class_names = class_data['classes']
                    else:
                        self.class_names = list(class_data.values())
                logger.info("✅ Class names loaded from JSON")
            else:
                # Fall back to label encoder classes
                self.class_names = self.label_encoder.classes_.tolist()
                logger.info("✅ Class names loaded from label encoder")
            
            # Load model info if available
            model_timestamp = model_path.name.split('_')[-1].replace('.pkl', '')
            info_files = list(self.model_dir.glob(f"model_info_{model_timestamp}.json"))
            if info_files:
                with open(info_files[0], 'r') as f:
                    self.model_info = json.load(f)
                logger.info("✅ Model info loaded")
            
            self.is_loaded = True
            logger.info(f"🎉 Model loading complete!")
            logger.info(f"Available gestures: {self.class_names}")
            logger.info(f"Number of classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(f"Model directory contents: {[f.name for f in self.model_dir.glob('*')]}")
            self.is_loaded = False
            raise
    
    def predict(self, landmarks: List[LandmarkPoint]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict gesture from hand landmarks.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # CHANGED: Extract and preprocess coordinates using XYZ (63 features)
            coords = extract_xyz_coords(landmarks)
            
            # Reshape for prediction (1 sample)
            X = coords.reshape(1, -1)
            
            logger.info(f"Input shape: {X.shape}, Expected features: 63")
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            predicted_gesture = self.label_encoder.inverse_transform([predicted_idx])[0]
            confidence = float(probabilities[predicted_idx])
            
            # Create probability dictionary
            prob_dict = {
                gesture: float(prob) 
                for gesture, prob in zip(self.class_names, probabilities)
            }
            
            return predicted_gesture, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "model_type": "XGBoost",
            "num_classes": len(self.class_names),
            "classes": self.class_names,
            "feature_dim": 63  # CHANGED: 21 landmarks * 3 coordinates (x, y, z)
        }
        
        # Add additional info if available
        if self.model_info:
            info.update(self.model_info)
            
        return info
