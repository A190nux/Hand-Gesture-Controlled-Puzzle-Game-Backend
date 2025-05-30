import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import json

API_BASE_URL = "http://localhost:8000"

def generate_random_landmarks():
    """Generate random hand landmarks for testing."""
    landmarks = []
    for i in range(21):
        landmarks.append({
            "x": random.uniform(0.1, 0.9),
            "y": random.uniform(0.1, 0.9),
            "z": random.uniform(-0.1, 0.1)
        })
    return landmarks

def make_prediction_request():
    """Make a single prediction request."""
    try:
        landmarks = generate_random_landmarks()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"landmarks": landmarks},
            timeout=5
        )
        return response.status_code, response.elapsed.total_seconds()
    except Exception as e:
        print(f"Request failed: {e}")
        return None, None

def load_test(duration_seconds=300, requests_per_second=2):
    """Run load test for specified duration."""
    print(f"Starting load test for {duration_seconds} seconds at {requests_per_second} RPS")
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        while time.time() - start_time < duration_seconds:
            # Submit requests
            futures = []
            for _ in range(requests_per_second):
                future = executor.submit(make_prediction_request)
                futures.append(future)
            
            # Wait for requests to complete
            for future in futures:
                status_code, response_time = future.result()
                request_count += 1
                if status_code == 200:
                    success_count += 1
                
                if request_count % 10 == 0:
                    print(f"Requests: {request_count}, Success: {success_count}, "
                          f"Success Rate: {success_count/request_count*100:.1f}%")
            
            # Wait before next batch
            time.sleep(1)
    
    print(f"\nLoad test completed!")
    print(f"Total requests: {request_count}")
    print(f"Successful requests: {success_count}")
    print(f"Success rate: {success_count/request_count*100:.1f}%")

if __name__ == "__main__":
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("API is ready!")
                break
        except:
            pass
        time.sleep(2)
    
    # Run load test
    load_test(duration_seconds=180, requests_per_second=3)