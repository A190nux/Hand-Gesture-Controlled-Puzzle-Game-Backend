import pandas as pd
import requests
import json
import time
from typing import List, Dict
import numpy as np

API_BASE_URL = "http://0.0.0.0:8000"

def load_test_data(csv_path: str = "tests/hand_landmarks_data.csv") -> pd.DataFrame:
    """Load the real hand landmarks data."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} samples from {csv_path}")
        print(f"📊 Classes: {df['label'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def convert_row_to_landmarks(row: pd.Series) -> List[Dict]:
    """Convert a CSV row to landmarks format for API."""
    landmarks = []
    
    # Extract coordinates for each of the 21 landmarks
    for i in range(1, 22):  # 1-based indexing in CSV
        x_col = f'x{i}'
        y_col = f'y{i}'
        z_col = f'z{i}'
        
        landmarks.append({
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "z": float(row[z_col])
        })
    
    return landmarks

def test_single_prediction(landmarks: List[Dict], true_label: str) -> Dict:
    """Test a single prediction."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"landmarks": landmarks},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted_label = result['gesture']
            confidence = result['confidence']
            processing_time = result['processing_time_ms']
            
            is_correct = predicted_label == true_label
            
            return {
                'success': True,
                'predicted': predicted_label,
                'true_label': true_label,
                'correct': is_correct,
                'confidence': confidence,
                'processing_time': processing_time,
                'probabilities': result['probabilities']
            }
        else:
            return {
                'success': False,
                'error': response.text,
                'status_code': response.status_code,
                'true_label': true_label
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'true_label': true_label
        }

def run_accuracy_test(df: pd.DataFrame, max_samples: int = 100) -> Dict:
    """Run accuracy test on real data."""
    print(f"🧪 Testing accuracy on {min(len(df), max_samples)} samples...")
    
    # Sample data if too large
    if len(df) > max_samples:
        df_sample = df.sample(n=max_samples, random_state=42)
        print(f"📝 Sampled {max_samples} from {len(df)} total samples")
    else:
        df_sample = df
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    processing_times = []
    
    for idx, row in df_sample.iterrows():
        landmarks = convert_row_to_landmarks(row)
        true_label = row['label']
        
        result = test_single_prediction(landmarks, true_label)
        results.append(result)
        
        if result['success']:
            total_predictions += 1
            if result['correct']:
                correct_predictions += 1
            processing_times.append(result['processing_time'])
            
            # Print progress every 10 samples
            if total_predictions % 10 == 0:
                current_accuracy = correct_predictions / total_predictions * 100
                print(f"Progress: {total_predictions}/{len(df_sample)} - "
                      f"Accuracy: {current_accuracy:.1f}% - "
                      f"Avg time: {np.mean(processing_times):.2f}ms")
        else:
            print(f"❌ Failed prediction for sample {idx}: {result.get('error', 'Unknown error')}")
    
    # Calculate final metrics
    accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    # Per-class accuracy
    class_results = {}
    for result in results:
        if result['success']:
            true_label = result['true_label']
            if true_label not in class_results:
                class_results[true_label] = {'correct': 0, 'total': 0}
            class_results[true_label]['total'] += 1
            if result['correct']:
                class_results[true_label]['correct'] += 1
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for label, stats in class_results.items():
        class_accuracy[label] = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    
    return {
        'overall_accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'failed_requests': len(df_sample) - total_predictions,
        'avg_processing_time_ms': avg_processing_time,
        'class_accuracy': class_accuracy,
        'class_results': class_results,
        'all_results': results
    }

def print_detailed_results(results: Dict):
    """Print detailed test results."""
    print("\n" + "="*60)
    print("🎯 ACCURACY TEST RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Correct Predictions: {results['correct_predictions']}/{results['total_predictions']}")
    print(f"Failed Requests: {results['failed_requests']}")
    print(f"Average Processing Time: {results['avg_processing_time_ms']:.2f}ms")
    
    print("\n📊 Per-Class Accuracy:")
    for label, accuracy in results['class_accuracy'].items():
        stats = results['class_results'][label]
        print(f"  {label}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    print("\n🔍 Sample Predictions:")
    for i, result in enumerate(results['all_results'][:5]):  # Show first 5
        if result['success']:
            status = "✅" if result['correct'] else "❌"
            print(f"  {status} True: {result['true_label']} | "
                  f"Predicted: {result['predicted']} | "
                  f"Confidence: {result['confidence']:.3f}")

def main():
    """Main test function."""
    print("🚀 Starting Real Data Accuracy Test")
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('model_loaded', False):
                    print("✅ API is ready and model is loaded!")
                    break
                else:
                    print("⚠️  API is up but model not loaded yet...")
            else:
                print(f"⚠️  API health check failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️  API not ready yet: {e}")
        
        if i == max_retries - 1:
            print("❌ API failed to become ready")
            return
        
        time.sleep(2)
    
    try:
        # Load test data
        df = load_test_data()
        
        # Run accuracy test
        results = run_accuracy_test(df, max_samples=50)  # Test on 50 samples
        
        # Print results
        print_detailed_results(results)
        
        # Save results to /tmp (writable location)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"/tmp/accuracy_test_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()