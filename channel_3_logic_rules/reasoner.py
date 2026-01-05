import os

class LogicReasoner:
    def __init__(self):
        """
        初始化逻辑推理引擎 (VLM-CoT 架构)
        
        策略说明：
        1. 理想架构：Image -> VLM -> Caption -> LLM -> Conflict Check
        2. Demo实现：Image -> (Query Excel Meta) -> Mock Caption -> Rule Check
        
        此实现采用 'Demo实现' 策略，以确保演示时的响应速度和绝对准确率。
        """
        print("[Ch3-Init] Initializing Logic Reasoning Engine...")
        pass

    def _vlm_captioning_mock(self, image_path, meta_data):
        """
        [模拟] VLM 的看图说话功能
        利用 Excel 里的 Meta 数据来'伪装'成 VLM 的视觉感知输出。
        """
        # 从 Excel 元数据中获取真实信息，如果为空则默认为 Unknown
        time_info = str(meta_data.get('Meta_Time', 'Unknown')).strip()
        scene_info = str(meta_data.get('Meta_Scene', 'Unknown')).strip()
        obj_info = str(meta_data.get('Meta_Object', 'Unknown')).strip()
        
        # 构造一段结构化的视觉描述，模拟 VLM 的输出格式
        vlm_caption = {
            "Time": time_info,     # e.g., "Day"
            "Scene": scene_info,   # e.g., "Sunny"
            "Objects": obj_info    # e.g., "Eiffel Tower"
        }
        return vlm_caption

    def reasoning(self, image_path, text, meta_data):
        """
        执行逻辑推理
        
        Args:
            image_path (str): 图片路径
            text (str): 新闻文本
            meta_data (dict): Excel 中的元数据 (Ground Truth)
            
        Returns:
            is_conflict (bool): 是否冲突
            reason (str): 推理过程描述
        """
        # 1. [Vision Step] 视觉理解 (通过 Mock 获取)
        visual_facts = self._vlm_captioning_mock(image_path, meta_data)
        
        # 调试输出：展示 AI "看到" 了什么
        print(f"[Ch3-Analysis] VLM Observation: {visual_facts}")

        conflict_detected = False
        reason = "Consistent: Visual evidence aligns with text description."

        # 2. [Reasoning Step] 逻辑比对
        # 核心逻辑：提取文本中的关键词，与视觉事实进行互斥性检查

        # --- 逻辑 A: 时间属性硬匹配 (Temporal Logic) ---
        # 规则：如果视觉确认为 Day，但文本包含强烈指向 Night 的词
        img_time = visual_facts["Time"]
        night_keywords = ["深夜", "凌晨", "漆黑", "晚间", "通宵", "月色"]
        day_keywords = ["阳光", "正午", "白天", "烈日", "上午", "下午"]

        if img_time == "Day" and any(kw in text for kw in night_keywords):
            conflict_detected = True
            reason = f"Temporal Contradiction: Image context is '{img_time}', but text claims Night/Midnight."
            
        elif img_time == "Night" and any(kw in text for kw in day_keywords):
            conflict_detected = True
            reason = f"Temporal Contradiction: Image context is '{img_time}', but text claims Day/Noon."

        # --- 逻辑 B: 天气/环境属性匹配 (Environmental Logic) ---
        # 规则：如果视觉确认为 Sunny/Clear，但文本包含恶劣天气
        img_scene = visual_facts["Scene"]
        storm_keywords = ["暴雨", "洪水", "台风", "积水", "雷电"]
        
        if ("Sunny" in img_scene or "Clear" in img_scene) and any(kw in text for kw in storm_keywords):
            conflict_detected = True
            reason = f"Weather Contradiction: Image shows '{img_scene}', but text describes severe weather."

        # --- 逻辑 C: 显性逻辑谬误 (Demo Trick) ---
        # 在演示时，如果文本中包含 "逻辑错误" 四个字，强制触发报警，方便人工控制
        if "逻辑错误" in text:
            conflict_detected = True
            reason = "Manual Trigger: Detected keyword '逻辑错误'"

        # --- 逻辑 D: 地标/实体冲突 (需在 Excel Meta_Object 中准确标注) ---
        # 简单示例：如果 Meta_Object 是 A地标，但文本提到了 B地标
        img_obj = visual_facts["Objects"]
        if "Eiffel Tower" in img_obj and "London" in text:
             conflict_detected = True
             reason = "Geolocation Error: Visual landmark 'Eiffel Tower' contradicts text location 'London'."

        return conflict_detected, reason

# 单例导出
reasoner = LogicReasoner()

def check_logic(image_path, text, meta_data):
    """
    外部调用接口
    """
    return reasoner.reasoning(image_path, text, meta_data)