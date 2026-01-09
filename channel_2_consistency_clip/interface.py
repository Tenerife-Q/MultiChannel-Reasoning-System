
import sys
import os
# Ensure we can find the sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from matcher import check_consistency
except ImportError:
    # Fallback if run from root
    from channel_2_consistency_clip.matcher import check_consistency

def get_ch2_score(image_path, text):
    try:
        score, status = check_consistency(image_path, text)
        return score
    except Exception as e:
        print(f"Ch2 Error: {e}")
        return 0.3 # Default safe score
