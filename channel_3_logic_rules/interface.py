
import os
import sys
import torch
import re
from PIL import Image
from transformers import AutoTokenizer

# Config
# Dynamic Path Calculation
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming structure: .../channel_3_logic_rules/interface.py
# Models in: .../channel_3_logic_rules/moondream2/vikhyatk/moondream2
base_repo_path = os.path.join(CURRENT_DIR, "moondream2", "vikhyatk")
MODEL_PATH = os.path.join(base_repo_path, "moondream2")

sys.path.append(base_repo_path)

# Lazy loading global variables
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    if model is not None: return

    try:
        from moondream2.hf_moondream import HfMoondream
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Ch3] Loading Moondream on {device}...")
        
        model = HfMoondream.from_pretrained(
            MODEL_PATH, 
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device}
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model.eval()
        print("[Ch3] Model loaded.")
    except Exception as e:
        print(f"[Ch3] Init Error: {e}")

def get_logic_score(image_path, text):
    init_model()
    if model is None: return 0.5 # Fail safe

    try:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        question = f"Rate how much the image matches the text '{text}' on a scale from 0 to 10, where 0 is no match and 10 is perfect match. Answer with a single number."
        answer = model.answer_question(enc_image, question, tokenizer)
        
        # Parse
        nums = re.findall(r"\d+\.?\d*", answer)
        if nums:
            val = float(nums[0])
            normalized = val / 10.0
            if normalized > 1.0: normalized = 1.0
            return 1.0 - normalized # Return Logic Conflict Score
        
        # Fallback
        if "yes" in answer.lower(): return 0.1
        if "no" in answer.lower(): return 0.9
        
    except Exception as e:
        print(f"[Ch3] Inference Error: {e}")
    
    return 0.5
