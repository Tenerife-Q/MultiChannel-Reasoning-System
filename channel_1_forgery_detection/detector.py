"""
================================================================================
Channel 1: Omni-Forgery Detection (全谱系篡改检测)
文件名: detector.py
定位: 整个系统的"物理防线"

================================================================================
【核心任务】
只要这张图的像素被动过手脚（不管是人P的，还是AI涂的），都要把它抓出来。
检验对象: 图片的微观痕迹（噪声分布、边缘特性），而非图片内容。

【检测目标 - 三种篡改类型】
1. 拼接/复制 (Splicing/Copy-Move)
   - 传统手段：把A图的人扣到B图，或把云彩复制一份
   - 特征：边缘像素突变

2. AI消除/重绘 (AI Inpainting)
   - 现代手段：用手机"消除笔"把路人涂掉，或用PS"创成式填充"补背景
   - 特征：噪声不匹配，AI生成区域过于"平滑"或带有特定棋盘格纹理

3. 压缩伪影 (Compression Artifacts)
   - 修改过的区域，其JPEG压缩痕迹与原图不同

================================================================================
【I/O 接口规范】
================================================================================
输入 (Input):
  - image_path (str): 图片在本地的路径
  - 示例: "./data/images/aigc_eraser_001.jpg"

输出 (Output):
  - score (float): 篡改置信度 P1，范围 0.0 ~ 1.0
    对应 Excel 字段: GT_Ch1_Tamper (1=有篡改, 0=无篡改)
  - message (str): 诊断结果描述

【与Excel字段的对应关系】
  - Sample_Type = "Tamper_PS"   -> 检测传统PS篡改 (Splicing/Clone)
  - Sample_Type = "Tamper_AIGC" -> 检测AI消除/重绘 (Inpainting)
  - Sample_Type = "Real"        -> 应输出低分 (无篡改)
  - Tamper_Method 字段记录具体手段: MagicEraser, Photoshop_Splicing 等

================================================================================
【技术选型】
推荐模型 (二选一):
  1. MVSS-Net++ (首选) - 对噪声和边缘非常敏感，能抓AI Inpainting
  2. MantraNet (备选) - 经典通用检测器，稳定性好

================================================================================
"""

import os
import random
# import torch  # TODO: 待模型集成时解开注释

class ForgeryDetector:
    """
    通用篡改检测器
    能同时兼容传统PS篡改和AI Inpainting篡改
    """
    
    def __init__(self):
        """
        初始化检测器
        任务: 加载 MVSS-Net 或 MantraNet 的预训练权重
        路径注意: 权重文件应存放在 channel_1_forgery_detection/weights/ 目录下
        """
        print("[Ch1-Init] Loading Omni-Forgery Detection Model...")
        print("[Ch1-Init] Target: MVSS-Net/MantraNet for PS + AIGC detection")
        # TODO: 实例化模型并加载权重
        # self.model = MVSS_Net()
        # model_path = os.path.join(os.path.dirname(__file__), 'weights/mvss_net.pth')
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        self.model_loaded = False  # 标记模型是否已加载

    def detect(self, image_path):
        """
        统一接口: 检测图片是否被篡改 (包含PS和AI修改)
        
        Args:
            image_path (str): 图片路径 (e.g., "./data/images/aigc_eraser_001.jpg")
        
        Returns:
            score (float): 篡改概率 P1，对应 GT_Ch1_Tamper
                           1.0 = 确定篡改, 0.0 = 确定真实
            message (str): 诊断结果描述
        
        对应 Excel 字段:
            - 输入: Image_Path (B列)
            - 验证: GT_Ch1_Tamper (F列), 1=有篡改, 0=无篡改
            - 分类: Sample_Type (B列), Tamper_PS / Tamper_AIGC / Real
        """
        # 1. 路径标准化处理 (防止相对路径找不到文件)
        if not os.path.exists(image_path):
            abs_path = os.path.abspath(image_path)
            if not os.path.exists(abs_path):
                return 0.0, f"[ERROR] File Not Found: {image_path}"
            image_path = abs_path

        file_name = os.path.basename(image_path)
        print(f"[Ch1-Analysis] Processing: {file_name}")
        print(f"[Ch1-Analysis] Analyzing noise distribution & edge features...")

        # -----------------------------------------------------------
        # TODO: [Phase 2] 接入真实模型推理
        # -----------------------------------------------------------
        # img_tensor = preprocess(image_path)
        # with torch.no_grad():
        #     pred_mask = self.model(img_tensor)
        #     score = pred_mask.max().item()
        #     # 保存 Mask 用于可视化演示
        #     mask_path = self._save_mask(pred_mask, image_path)
        # return score, f"Tamper probability: {score:.4f}"
        # -----------------------------------------------------------

        # ============================================================
        # [Phase 1] Mock Logic - 基于文件名/路径模拟检测结果
        # 用于联调测试，真实模型接入前的演示
        # ============================================================
        return self._mock_detect(image_path)

    def _mock_detect(self, image_path):
        """
        Mock 检测逻辑
        根据文件名关键词模拟不同的检测结果
        
        对应 Sample_Type 分类:
          - Tamper_AIGC: aigc_, inpaint, eraser, remove, magic
          - Tamper_PS: ps_, splicing, copymove, clone, tamper, fake
          - Real: real_, 或不含上述关键词
        
        对应 Tamper_Method:
          - MagicEraser, Photoshop_Splicing, Photoshop_Clone 等
        """
        file_name = image_path.lower()
        file_basename = os.path.basename(file_name)
        
        # -----------------------------------------------------------------
        # Case 1: AIGC 篡改检测 (AI Inpainting/Eraser)
        # 对应 Sample_Type = "Tamper_AIGC", GT_Ch1_Tamper = 1
        # Tamper_Method: MagicEraser, AI_Inpaint 等
        # -----------------------------------------------------------------
        aigc_keywords = ["aigc", "inpaint", "eraser", "remove", "magic"]
        if any(kw in file_name for kw in aigc_keywords):
            # 模拟高置信度检测 (0.88 ~ 0.98)
            score = round(random.uniform(0.88, 0.98), 4)
            
            # 细分 AIGC 篡改类型
            if "eraser" in file_name or "remove" in file_name:
                method = "AI Object Removal (Eraser)"
            elif "inpaint" in file_name:
                method = "AI Inpainting (Texture Synthesis)"
            else:
                method = "AIGC Modification"
            
            msg = f"[DETECTED] {method} - Noise anomaly in smooth regions"
            print(f"[Ch1-Result] Score={score:.4f}, Type=AIGC")
            return score, msg
        
        # -----------------------------------------------------------------
        # Case 2: 传统 PS 篡改检测 (Splicing/Copy-Move/Clone)
        # 对应 Sample_Type = "Tamper_PS", GT_Ch1_Tamper = 1
        # Tamper_Method: Photoshop_Splicing, Photoshop_Clone 等
        # -----------------------------------------------------------------
        ps_keywords = ["ps_", "splicing", "copymove", "clone", "tamper", "fake"]
        if any(kw in file_name for kw in ps_keywords):
            # 模拟高置信度检测 (0.85 ~ 0.96)
            score = round(random.uniform(0.85, 0.96), 4)
            
            # 细分 PS 篡改类型
            if "splicing" in file_name:
                method = "Image Splicing (Edge discontinuity)"
            elif "copymove" in file_name or "clone" in file_name:
                method = "Copy-Move Forgery (Duplicated regions)"
            else:
                method = "Traditional PS Manipulation"
            
            msg = f"[DETECTED] {method} - Edge artifacts detected"
            print(f"[Ch1-Result] Score={score:.4f}, Type=PS")
            return score, msg
        
        # -----------------------------------------------------------------
        # Case 3: 真实图片 (无篡改)
        # 对应 Sample_Type = "Real" 或 "Mismatch" 或 "Logic_Trap"
        # GT_Ch1_Tamper = 0
        # -----------------------------------------------------------------
        # 模拟低置信度 (0.02 ~ 0.12)
        score = round(random.uniform(0.02, 0.12), 4)
        msg = "[CLEAN] No manipulation detected - Noise pattern consistent"
        print(f"[Ch1-Result] Score={score:.4f}, Type=Real")
        return score, msg

    def _save_mask(self, mask_tensor, original_path):
        """
        保存篡改掩码图 (Mask)
        白色区域表示被篡改的位置
        
        TODO: 真实模型接入时实现
        """
        # mask_dir = os.path.join(os.path.dirname(__file__), 'output_masks')
        # os.makedirs(mask_dir, exist_ok=True)
        # mask_filename = os.path.basename(original_path).replace('.jpg', '_mask.png')
        # mask_path = os.path.join(mask_dir, mask_filename)
        # save_image(mask_tensor, mask_path)
        # return mask_path
        pass


# ============================================================================
# 单例模式导出
# ============================================================================
detector = ForgeryDetector()


def detect_tamper(image_path):
    """
    外部调用接口 (标准函数)
    供 main.py 或其他模块调用
    
    Args:
        image_path (str): 图片路径
    
    Returns:
        tuple: (score, message)
            - score (float): 篡改概率 P1 (0.0 ~ 1.0)
            - message (str): 诊断结果
    
    使用示例:
        from channel_1_forgery_detection.detector import detect_tamper
        score, msg = detect_tamper("data/images/aigc_eraser_001.jpg")
        print(f"P1 = {score}, Result: {msg}")
    """
    return detector.detect(image_path)


def detect_tamper_pipeline(image_path):
    """
    Pipeline 接口 (别名)
    兼容不同的调用方式
    """
    return detector.detect(image_path)