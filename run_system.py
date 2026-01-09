import os
import sys
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# --- Path Config ---
# Make path dynamic for portability (Docker/Different PCs)
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(WORKSPACE_ROOT, "data", "Yuanjing_Data_Standard_Ultimate_Final.xlsx")
REPORT_FILE = os.path.join(WORKSPACE_ROOT, "dataset", "Final_Vote_Report.csv")

# Add workspace to path
sys.path.append(WORKSPACE_ROOT)

# --- Import Interfaces ---
# We use dynamic imports or safe imports to handle missing dependencies gracefully
print("Initializing System Interfaces...")

try:
    from channel_1_forgery_detection.interface import get_ch1_detector
    ch1_weight_path = os.path.join(WORKSPACE_ROOT, "channel_1_forgery_detection", "weight", "mvssnet_casia.pth")
    ch1_detector = get_ch1_detector_obj = get_ch1_detector(weight_path=ch1_weight_path) # returns object with .detect()
    print("[Main] Channel 1 Interface Loaded.")
except Exception as e:
    print(f"[Main] Warning: Channel 1 interface failed ({e}). Using Mock.")
    ch1_detector = None

try:
    from channel_2_consistency_clip.interface import get_ch2_score
    print("[Main] Channel 2 Interface Loaded.")
except Exception as e:
    print(f"[Main] Warning: Channel 2 interface failed ({e}). Using Mock.")
    get_ch2_score = None

try:
    from channel_3_logic_rules.interface import get_logic_score
    print("[Main] Channel 3 Interface Loaded.")
except Exception as e:
    print(f"[Main] Warning: Channel 3 interface failed ({e}). Using Mock.")
    get_logic_score = None


def resolve_image_path(rel_path):
    """Helper to find image path robustly"""
    if pd.isna(rel_path): return None
    rel_path = str(rel_path).replace("\\", "/")
    
    # Check 1: Absolute or relative to CWD
    p1 = os.path.normpath(os.path.join(WORKSPACE_ROOT, rel_path))
    if os.path.exists(p1): return p1
    
    # Check 2: data/images prefix logic
    if rel_path.startswith("data/"):
        p2 = os.path.join(WORKSPACE_ROOT, rel_path.replace("data/", ""))
        if os.path.exists(p2): return p2

    # Check 3: Flat search
    filename = os.path.basename(rel_path)
    base = os.path.join(WORKSPACE_ROOT, "data", "images")
    p3 = os.path.join(base, filename)
    if os.path.exists(p3): return p3
    
    return None


def run_full_inference():
    print("\n========================================================")
    print("      Multi-Channel Forensic System - Live Mode      ")
    print("========================================================")
    
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return

    df = pd.read_excel(DATA_FILE)
    print(f"Loaded {len(df)} samples.")
    
    results = []
    
    # Define thresholds
    TH_CH1 = 0.5  # > 0.5 Fake
    TH_CH2 = 0.22 # < 0.22 Fake
    TH_CH3 = 0.5  # > 0.5 Fake

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_path = resolve_image_path(row['Image_Path'])
        text = str(row['Text_Content'])
        
        # --- Channel 1: Tamper Detection ---
        s1 = 0.0
        if image_path and ch1_detector:
            try:
                det_ret = ch1_detector.detect(image_path)
                if isinstance(det_ret, tuple):
                    s1 = float(det_ret[0])
                else:
                    s1 = float(det_ret)
            except Exception as e:
                print(f"Error in Ch1 detect: {e}")
                s1 = 0.5
        else:
            s1 = 0.0 # Default Safe
        # --- Channel 2: Semantic Consistency ---
        s2_raw = 0.3
        if image_path and get_ch2_score:
            s2_raw = get_ch2_score(image_path, text)
            
        # --- Channel 3: Logic Reasoning ---
        s3 = 0.0
        if image_path and get_logic_score:
            s3 = get_logic_score(image_path, text)
            
        # --- Voting Logic ---
        reasons = []
        is_fake = False
        max_risk = 0.0
        
        # Check Ch1
        if s1 > TH_CH1:
            is_fake = True
            reasons.append(f"Ch1:Physical Tamper({s1:.2f})")
            max_risk = max(max_risk, s1)
            
        # Check Ch2 (Low Score = Fake)
        if s2_raw < TH_CH2:
            is_fake = True
            reasons.append(f"Ch2:Semantic Mismatch({s2_raw:.2f})")
            max_risk = max(max_risk, 0.9)
        
        # Check Ch3
        if s3 > TH_CH3:
            is_fake = True
            reasons.append(f"Ch3:Logic Conflict({s3:.2f})")
            max_risk = max(max_risk, s3)
            
        final_verdict = 1 if is_fake else 0
        final_reason = " | ".join(reasons) if reasons else "Consistent"
        
        results.append({
            "ID": row.get('ID', idx),
            "Image_Path": row['Image_Path'],
            "GT_Label": row.get('GT_Final_Label'),
            "Pred_Label": final_verdict,
            "Risk_Score": float(max_risk),
            "Reason": final_reason,
            "Score_Ch1": round(s1, 3),
            "Score_Ch2": round(s2_raw, 3),
            "Score_Ch3": round(s3, 3)
        })

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(REPORT_FILE, index=False)
    
    # Simple Accuracy Metric
    valid = out_df.dropna(subset=['GT_Label'])
    if len(valid) > 0:
        acc = (valid['GT_Label'] == valid['Pred_Label']).sum() / len(valid)
        print(f"\nOptimization Completed.")
        print(f"System Correctness: {acc*100:.2f}%")
        
    print(f"Report saved to: {REPORT_FILE}")

if __name__ == "__main__":
    run_full_inference()
