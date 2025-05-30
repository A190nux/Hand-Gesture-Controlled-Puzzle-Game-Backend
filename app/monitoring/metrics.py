from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import List
import numpy as np
from app.schemas.request_response import LandmarkPoint

# Prediction time metric
prediction_time_histogram = Histogram(
    'gesture_prediction_duration_seconds',
    'Time spent on gesture prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Request counter for RPM calculation
request_counter = Counter(
    'gesture_prediction_requests_total',
    'Total number of gesture prediction requests'
)

# Request rate gauge (requests per minute)
requests_per_minute = Gauge(
    'gesture_prediction_requests_per_minute',
    'Number of requests per minute'
)

# Landmark coordinate statistics
landmark_x_avg = Gauge(
    'landmark_coordinates_x_average',
    'Average X coordinate of landmarks'
)

landmark_y_avg = Gauge(
    'landmark_coordinates_y_average',
    'Average Y coordinate of landmarks'
)

landmark_z_avg = Gauge(
    'landmark_coordinates_z_average',
    'Average Z coordinate of landmarks'
)

# Prediction confidence
prediction_confidence = Histogram(
    'gesture_prediction_confidence',
    'Confidence score of gesture predictions',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Gesture distribution
gesture_predictions = Counter(
    'gesture_predictions_by_class',
    'Number of predictions by gesture class',
    ['gesture_class']
)

# System health metrics
model_load_status = Gauge(
    'model_loaded_status',
    'Whether the ML model is loaded (1) or not (0)'
)

class MetricsCollector:
    def __init__(self):
        self.request_times = []
        self.request_window_size = 60  # 1 minute window
        
    def record_prediction_time(self, duration_seconds: float):
        """Record prediction time."""
        prediction_time_histogram.observe(duration_seconds)
    
    def record_request(self):
        """Record a new request."""
        request_counter.inc()
        current_time = time.time()
        self.request_times.append(current_time)
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Update requests per minute
        requests_per_minute.set(len(self.request_times))
    
    def record_landmark_stats(self, landmarks: List[LandmarkPoint]):
        """Record average landmark coordinates."""
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        z_coords = [landmark.z for landmark in landmarks]
        
        landmark_x_avg.set(np.mean(x_coords))
        landmark_y_avg.set(np.mean(y_coords))
        landmark_z_avg.set(np.mean(z_coords))
    
    def record_prediction_result(self, gesture: str, confidence: float):
        """Record prediction results."""
        prediction_confidence.observe(confidence)
        gesture_predictions.labels(gesture_class=gesture).inc()
    
    def update_model_status(self, is_loaded: bool):
        """Update model load status."""
        model_load_status.set(1 if is_loaded else 0)

# Global metrics collector instance
metrics_collector = MetricsCollector()
