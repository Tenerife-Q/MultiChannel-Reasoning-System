import os
import random
# import torch  # TODO: 待模型集成时解开注释

class ForgeryDetector:
    def __init__(self):
        """
        初始化检测器
        任务：加载 MVSS-Net 或 MantraNet 的预训练权重。
        路径注意：权重文件应存放在 channel_1_forgery/weights/ 目录下。
        """
        print("[Ch1-Init] Loading Forgery Detection Model...")
        # TODO: 实例化模型并加载权重
        # self.model = MVSS_Net()
        # model_path = os.path.join(os.path.dirname(__file__), 'weights/mvss_net.pth')
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        pass

    def detect(self, image_path):
        """
        执行检测
        Args:
            image_path (str): 图片相对路径 (e.g., "./data/images/001.jpg")
        Returns:
            score (float): 篡改概率
            message (str): 诊断结果
        """
        # 1. 路径标准化处理 (防止相对路径找不到文件)
        # 假设 image_path 是相对于项目根目录的
        if not os.path.exists(image_path):
            # 尝试拼接绝对路径
            abs_path = os.path.abspath(image_path)
            if not os.path.exists(abs_path):
                return 0.0, f"Error: File Not Found at {image_path}"
            image_path = abs_path

        print(f"[Ch1-Analysis] Processing: {os.path.basename(image_path)}")

        # -----------------------------------------------------------
        # TODO: 接入真实模型推理
        # img_tensor = preprocess(image_path)
        # with torch.no_grad():
        #     pred_mask = self.model(img_tensor)
        #     score = pred_mask.max().item()
        #     # save_mask(pred_mask, output_path) # 记得保存 Mask 用于演示
        # -----------------------------------------------------------

        # === Mock Logic (仅用于联调测试，基于文件名判断) ===
        file_name = image_path.lower()
        
        # 模拟 AI 篡改检测 (Inpainting)
        # 对应 Excel 中 Tamper_Type = AIGC 的数据
        if "aigc" in file_name or "remove" in file_name or "inpaint" in file_name:
            return 0.92, "Detected Inpainting Artifacts (AIGC)"
            
        # 模拟 传统 PS 检测 (Splicing)
        # 对应 Excel 中 Tamper_Type = PS 的数据
        elif "ps" in file_name or "tamper" in file_name or "fake" in file_name:
            # 配合 CSV 中的 ID 001 (Ch1_Tamper_Label=1)
            return 0.95, "Detected Splicing Artifacts (PS)"
            
        # 模拟 真图
        else:
            return 0.05, "No manipulation detected"

# 单例导出
detector = ForgeryDetector()

def detect_tamper(image_path):
    """
    外部调用接口
    """
    return detector.detect(image_path)