import requests
import json

API_BASE_URL = "http://localhost:8000"

def generate_random_landmarks():
    """Generate random hand landmarks for testing."""
    landmarks = []
    for i in range(21):
        landmarks.append({
            "x": 0.5 + (i * 0.01),  # Values between 0.5 and 0.7
            "y": 0.5 + (i * 0.01),  # Values between 0.5 and 0.7
            "z": 0.0
        })
    return landmarks

def test_single_request():
    """Test a single request and print the response."""
    try:
        landmarks = generate_random_landmarks()
        print(f"Sending {len(landmarks)} landmarks")
        print(f"Sample landmark: {landmarks[0]}")
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"landmarks": landmarks},
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code != 200:
            print(f"Error details: {response.json()}")
            
    except Exception as e:
        print(f"Request failed with exception: {e}")

if __name__ == "__main__":
    test_single_request()