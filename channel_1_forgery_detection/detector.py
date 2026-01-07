import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from models.mvssnet import get_mvss

def cv2_imread(file_path):
    """
    Read image with non-ASCII path support.
    """
    try:
        # Read file as byte array
        img_array = np.fromfile(file_path, np.uint8)
        # Decode the image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return None

def cv2_imwrite(file_path, img):
    """
    Write image with non-ASCII path support.
    """
    try:
        # Encode image
        retval, buf = cv2.imencode(os.path.splitext(file_path)[1], img)
        if retval:
            # Write key to file
            with open(file_path, "wb") as f:
                buf.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error writing image {file_path}: {e}")
        return False

class ForgeryDetector:
    def __init__(self, weight_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Detector on {self.device}...")
        
        # Initialize Model Structure
        # MVSS-Net uses ResNet50 backbone. 
        # We set pretrained_base=False because we will load full weights from checkpoint.
        self.model = get_mvss(backbone='resnet50',
                              pretrained_base=False,
                              nclass=1,
                              sobel=True,
                              constrain=True,
                              n_input=3)
        
        # Load Weights
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
            
        print(f"Loading weights from {weight_path}...")
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            # Handle if checkpoint is wrapped in 'state_dict' or is the dict itself
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                self.model.load_state_dict(checkpoint, strict=True)
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define Transformations (Standard ImageNet Normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print("Detector ready.")

    def preprocess(self, img_bgr, resize_size=512):
        """
        Preprocess image for the model.
        Args:
            img_bgr: Input image in BGR format (OpenCV default).
            resize_size: Target size for inference.
        Returns:
            tensor: Preprocessed tensor (1, C, H, W)
            ori_size: (H, W) for restoring mask size.
        """
        if img_bgr is None:
            raise ValueError("Image is None")
            
        ori_h, ori_w = img_bgr.shape[:2]
        
        # Resize
        img_resized = cv2.resize(img_bgr, (resize_size, resize_size))
        
        # OpenCV (BGR) -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Transform (ToTensor + Normalize)
        # ToTensor converts [0, 255] -> [0.0, 1.0] and HWC -> CHW
        img_tensor = self.transform(img_rgb)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, (ori_h, ori_w)

    def detect(self, image_path, resize=512, threshold=0.5):
        """
        Detect forgery in an image.
        Args:
            image_path: Path to the image file.
            resize: Size to resize image for model input.
            threshold: (Optional) Threshold for binary mask, irrelevant for score calculation usually.
        Returns:
            score: (float) Suspicion score (0.0 - 1.0).
            mask: (np.array) Forgery probability map (0-255 uint8), resized to original image size.
        """
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return 0.0, None

        # Use custom imread for non-ASCII path support
        img = cv2_imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return 0.0, None

        try:
            input_tensor, ori_size = self.preprocess(img, resize_size=resize)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                # Forward pass
                outputs = self.model(input_tensor)
                
                # MVSSNet returns (edge_out, seg_out) or similar list
                if isinstance(outputs, (list, tuple)):
                    seg_logit = outputs[-1] 
                else:
                    seg_logit = outputs

                # Sigmoid for probability
                seg_prob = torch.sigmoid(seg_logit)
                
                # Move to CPU
                seg_prob_np = seg_prob.squeeze().cpu().numpy() # (H, W)

            # Calculate Score (Max probability in the map)
            score = float(np.max(seg_prob_np))

            # Resize Mask back to original size
            # Convert to 0-255 for resizing
            mask_uint8_small = (seg_prob_np * 255).astype(np.uint8)
            mask_final = cv2.resize(mask_uint8_small, (ori_size[1], ori_size[0]))
            
            return score, mask_final

        except Exception as e:
            print(f"Inference error for {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, None

def main():
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming weight is in ./weight/mvssnet_casia.pth
    weight_path = os.path.join(current_dir, "weight", "mvssnet_casia.pth")
    
    # Input Data Directory
    workspace_root = os.path.dirname(os.path.dirname(current_dir)) # E:\原镜
    # Adjust logic: current_dir is E:\原镜\MultiChannel-Reasoning-System\channel_1_forgery_detection
    workspace = os.path.dirname(current_dir) # E:\原镜\MultiChannel-Reasoning-System
    
    data_images_dir = os.path.join(workspace, "data", "images")
    output_dir = os.path.join(current_dir, "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("MVSS-Net Forgery Detector")
    print(f"Weights: {weight_path}")
    print(f"Input:   {data_images_dir}")
    print(f"Output:  {output_dir}")
    print("="*50)
    
    detector = ForgeryDetector(weight_path)
    
    # Get Images
    exts = ('.jpg', '.png', '.jpeg', '.tif', '.bmp')
    if not os.path.exists(data_images_dir):
        print(f"Error: Data directory not found: {data_images_dir}")
        return

    images = [f for f in os.listdir(data_images_dir) if f.lower().endswith(exts)]
    print(f"Found {len(images)} images.")
    
    results = []
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(data_images_dir, img_name)
        print(f"[{idx+1}/{len(images)}] Processing {img_name}...", end='\r')
        
        score, mask = detector.detect(img_path)
        
        if mask is not None:
            # Save Mask
            save_name = os.path.splitext(img_name)[0] + f"_score_{score:.3f}.png"
            output_path = os.path.join(output_dir, save_name)
            cv2_imwrite(output_path, mask)
            results.append((img_name, score))
        else:
            results.append((img_name, -1))
            
    print("\n\n" + "="*50)
    print("Summary:")
    print(f"{'Image Name':<30} | {'Score':<10} | {'Verdict'}")
    print("-" * 55)
    
    TH = 0.5 
    
    for name, score in results:
        verdict = "Fake" if score > TH else "Real"
        if score == -1: verdict = "Error"
        print(f"{name:<30} | {score:.4f}     | {verdict}")
        
    print("="*50)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
