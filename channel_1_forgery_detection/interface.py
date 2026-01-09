import os
import sys
import numpy as np

# Ensure the current directory is in sys.path so we can import detector and models
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class MockCh1Detector:
    def __init__(self):
        print("[Ch1] WARNING: Running in MOCK mode (Random Scores).")
        
    def detect(self, image_path):
        # Mock logic: return random score between 0 and 1
        return np.random.uniform(0.0, 1.0)

def get_ch1_detector(weight_path=None):
    """
    Factory function to return a ForgeryDetector instance.
    Falls back to MockCh1Detector if weights are missing or loading fails.
    """
    print(f"[Ch1-Init] Initializing Channel 1 Interface...")
    
    if weight_path and os.path.exists(weight_path):
        try:
            print(f"[Ch1-Init] Found weights at: {weight_path}")
            # Import here to avoid early import errors
            # We assume detector.py is in the same folder as this interface.py
            from detector import ForgeryDetector
            
            # Initializing real detector
            detector = ForgeryDetector(weight_path=weight_path)
            print("[Ch1-Init] MVSS-Net Model loaded successfully.")
            return detector
            
        except Exception as e:
            print(f"[Ch1-Init] Error loading MVSS-Net: {e}")
            print("[Ch1-Init] Fallback to Mock Detector.")
            return MockCh1Detector()
    else:
        print(f"[Ch1-Init] Warnings: Weights not found at: {weight_path}")
        return MockCh1Detector()
