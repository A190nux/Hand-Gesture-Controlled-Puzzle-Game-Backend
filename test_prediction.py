import requests
import json

# Test data - 21 landmarks with realistic coordinates
test_landmarks = []
for i in range(21):
    test_landmarks.append({
        "x": 0.5 + (i * 0.01),  # Vary x slightly
        "y": 0.5 + (i * 0.01),  # Vary y slightly  
        "z": 0.0
    })

# Test the API
url = "http://localhost:8000/predict"
data = {"landmarks": test_landmarks}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# Test other endpoints
print("\n--- Health Check ---")
response = requests.get("http://localhost:8000/health")
print(json.dumps(response.json(), indent=2))

print("\n--- Available Gestures ---")
response = requests.get("http://localhost:8000/gestures")
print(json.dumps(response.json(), indent=2))