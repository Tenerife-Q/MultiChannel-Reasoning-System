import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
import torch.nn.functional as F

class ConsistencyDetector:
    """
    【最终修正版】Channel 2: 语义一致性检测器
    核心机制: 
    1. 自动翻译 (中文 -> 英文)
    2. 计算纯净的余弦相似度 (Raw Cosine Similarity)
    3. 拒绝 Logit Scale 干扰
    """
    
    def __init__(self):
        print("[Ch2-Init] 正在加载 CLIP 模型 (openai/clip-vit-base-patch32)...")
        # 检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Ch2-Init] 运行设备: {self.device}")

        try:
            # 加载模型
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # 初始化翻译器
            self.translator = GoogleTranslator(source='auto', target='en')
            print("[Ch2-Init] 模型加载成功！纯净相似度计算模式已启动。")
        except Exception as e:
            print(f"[Ch2-Error] 初始化失败: {e}")

    def _translate(self, text):
        """将非英文文本翻译为英文"""
        try:
            # 简单的判断：如果包含中文(unicode范围)，则翻译
            if any(u'\u4e00' <= c <= u'\u9fff' for c in text):
                translated = self.translator.translate(text)
                print(f"    [翻译] {text[:10]}... -> {translated[:40]}...")
                return translated
            return text
        except Exception as e:
            print(f"    [Warn] 翻译服务超时/失败，使用原文: {e}")
            return text

    def check(self, image_path, text):
        if not os.path.exists(image_path):
            return 0.0, "[ERROR] File not found"

        try:
            # 1. 文本处理：翻译 + 截断
            text_en = self._translate(text)
            text_en = text_en[:77] # CLIP 长度限制

            # 2. 图像加载
            image = Image.open(image_path)

            # 3. 预处理 inputs
            inputs = self.processor(
                text=[text_en], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            # 4. 模型推理 (不计算梯度)
            with torch.no_grad():
                # 注意：这里我们不直接调 model(**inputs)，而是分别提取特征
                # 这样可以避开模型内部的 logit_scale (温度系数)
                
                # A. 获取图片特征向量
                image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                # B. 获取文本特征向量
                text_features = self.model.get_text_features(input_ids=inputs['input_ids'])

                # 5. 归一化 (Normalization) 
                # 将向量长度缩放为 1，这样点积(Dot Product)就等于余弦相似度
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                # 6. 计算余弦相似度 (Dot Product)
                # 形状变化: (1, 512) * (1, 512).T -> (1, 1)
                similarity = (image_features @ text_features.T).item()

            # 7. 判定逻辑 (基于纯净相似度的阈值)
            # -----------------------------------------------
            # 真实数据分布经验 (CLIP ViT-Base):
            # Same Semantics (熊猫vs熊猫): ~0.26 - 0.32
            # Different (熊猫vs老虎):      ~0.15 - 0.22
            # -----------------------------------------------
            # 我们设定阈值为 0.225 (这是一个非常灵敏的红线)
            
            THRESHOLD = 0.225

            if similarity < THRESHOLD:
                status = f"[MISMATCH] 语义冲突 (Sim={similarity:.4f} < {THRESHOLD})"
            else:
                status = f"[CONSISTENT] 语义一致 (Sim={similarity:.4f} > {THRESHOLD})"

            return similarity, status

        except Exception as e:
            print(f"[Error] 推理崩溃: {e}")
            return 0.0, str(e)

# 单例导出
detector = ConsistencyDetector()

def check_consistency(image_path, text, ground_truth=None):
    return detector.check(image_path, text)