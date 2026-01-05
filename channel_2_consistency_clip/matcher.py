import os
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import torch

class ConsistencyDetector:
    def __init__(self):
        """
        初始化检测器
        任务：加载 OpenAI CLIP 模型 (ViT-Base-Patch32)。
        注意：首次运行会自动下载模型权重 (~600MB)，请保持网络通畅。
        """
        print("[Ch2-Init] Loading CLIP Model (openai/clip-vit-base-patch32)...")
        
        # -----------------------------------------------------------
        # TODO: [Step 1] 集成真实 CLIP 模型
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # -----------------------------------------------------------
        pass

    def check(self, image_path, text):
        """
        执行图文一致性检测
        
        Args:
            image_path (str): 图片相对路径
            text (str): 待检测的文本内容
            
        Returns:
            score (float): 0.0 - 1.0 (匹配度)
        """
        # 1. 路径处理
        if not os.path.exists(image_path):
            # 尝试修复路径
            abs_path = os.path.abspath(image_path)
            if os.path.exists(abs_path):
                image_path = abs_path
            else:
                return 0.0 # 文件不存在，无法匹配

        # 文本截断处理 (CLIP通常限制77 token)
        short_text = text[:70] 

        print(f"[Ch2-Analysis] Matching Image '{os.path.basename(image_path)}' with Text: '{short_text}...'")

        # -------------------------------------------------------------
        # TODO: [Step 2] 接入真实 CLIP 推理
        # image = Image.open(image_path)
        # inputs = self.processor(text=[short_text], images=image, return_tensors="pt", padding=True).to(self.device)
        # 
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     logits_per_image = outputs.logits_per_image  # image-text similarity score
        #     probs = logits_per_image.softmax(dim=1)      # label probabilities
        #     
        #     # 简单的相似度归一化处理 (CLIP raw logits 需要 sigmoid 或 softmax 处理)
        #     # 这里建议直接取 logits_per_image.item() / 100 粗略归一化，或者使用 Cosine Similarity
        #     score = probs[0][0].item() 
        # -------------------------------------------------------------

        # =============================================================
        # [Mock Logic] 模拟逻辑 (用于联调)
        # 基于 Excel 中的逻辑构造：图文不符 (移花接木) vs 匹配
        # =============================================================
        
        file_name = image_path.lower()
        
        # 场景 A: 典型的图文不符 (移花接木)
        # 假设我们在 Excel 里故意放了一些 'fire' 的图，但配了 'sunny' 的文字
        # 或者 ID 为 002 (你的示例数据: Forest Fire 图 + 商场火灾文 -> 看起来相关其实不符)
        # 为了 Mock 效果，我们简单判定：如果 ID=002 或 Ch2_Consis_Label=0
        
        # 这里仅作简单的关键词反向匹配演示 Mock
        is_mismatch_mock = False
        
        # 模拟逻辑：如果图是 fire，文是 mall (Excel ID 002 case)
        if "fire" in file_name and "商场" in text:
            is_mismatch_mock = True
        # 模拟逻辑：如果图是 sunny，文是 台风 (Excel ID 003 case)
        elif "sunny" in file_name and "台风" in text:
            is_mismatch_mock = True
            
        if is_mismatch_mock:
            # 返回低分，表示不匹配
            return 0.15
        else:
            # 返回高分，表示匹配 (ID 001 case)
            return 0.88

# 单例导出
detector = ConsistencyDetector()

def check_consistency(image_path, text):
    """
    外部调用接口
    """
    return detector.check(image_path, text)