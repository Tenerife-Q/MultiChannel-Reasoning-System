"""
================================================================================
Channel 3: Logical Reasoning Engine (VLM-CoT) (逻辑与事实推理检测)
文件名: reasoner.py
定位: 系统逻辑层防线，处理CLIP无法识别的细粒度属性冲突与常识谬误

================================================================================
【核心任务】
构建具备"深度认知"能力的AI审判官，通过视觉大模型(VLM)与思维链(Chain of Thought)
技术，解决语义层(通道二)无法覆盖的细粒度逻辑冲突。

【与通道二(CLIP)的关键区别】
  通道二 (语义一致性): 解决 "Topic Alignment" (主题是否一致)
    - 能力边界: 只能判断"图和文是不是在说同一件事"
    - 对细节(天气、时间、数量)不敏感
  
  通道三 (逻辑推理): 解决 "Fact Verification" (事实是否冲突)
    - 能力边界: 在主题一致的前提下，通过VLM(视觉转译)和LLM(逻辑比对)
    - 寻找时空、因果、常识上的矛盾

【检测目标 - 三种冲突类型】
1. 细粒度属性冲突 (Fine-grained Attribute Conflict)
   - 时间: 图(正午) vs 文(深夜)
   - 天气: 图(晴天) vs 文(暴雨)
   - 数量: 图(空地) vs 文(人山人海)

2. 实体/地标错位 (Entity Mismatch)
   - 地标: 图(东方明珠) vs 文(东京塔)
   - 文字: 图中路牌/横幅文字与新闻内容矛盾 (OCR能力)

3. 常识因果谬误 (Common Sense Error)
   - 物理常识: 图(夏天短袖) vs 文(大雪纷飞)

【技术原理 - VLM-CoT (Visual Chain of Thought)】
  Step 1: 视觉转译 (Captioning)
    - 利用VLM将图片转化为结构化元数据
    - Prompt: "Describe focusing on: time of day, weather, location, quantity"
  
  Step 2: 逻辑比对 (Reasoning)
    - 利用LLM进行NLI(自然语言推理)任务
    - Logic: Premise(Image) <-> Hypothesis(Text) ?

================================================================================
【I/O 接口规范】
================================================================================
输入 (Input):
  - image_path (str): 图片路径
  - text (str): 新闻文本
  - meta_data (dict): Excel中的元数据行 (Mock模式下作为推理依据)

输出 (Output):
  - is_conflict (bool): True=逻辑冲突(Fake), False=逻辑自洽(Real)
    对应 Excel 字段: GT_Ch3_Logic (1=有冲突, 0=无冲突)
  - reason (str): 推理证据描述

【与Excel字段的对应关系】
  - Sample_Type = "Logic_Trap" -> 逻辑陷阱样本，GT_Ch3_Logic = 1
  - 输入: Image_Path (C列), Text_Content (D列)
  - 验证: GT_Ch3_Logic (H列), 1=有冲突, 0=无冲突
  - 元数据: Meta_Time, Meta_Weather, Meta_Location, Meta_Fact, Meta_Object

【Mock模式说明】
  鉴于演示环境算力限制，系统默认开启 Mock Mode (模拟模式)：
  - 通过读取预处理的元数据(Excel Ground Truth)模拟VLM的输出
  - 确保演示的低延迟与高准确率

================================================================================
【技术选型】
推荐模型:
  - VLM: Moondream (轻量级) 或 LLaVA (高精度)
  - LLM: 本地部署或API调用

================================================================================
"""

import os
import re
import pandas as pd
# import torch  # TODO: 待模型集成时解开注释


class LogicReasoner:
    """
    逻辑推理引擎 (VLM-CoT 架构)
    处理CLIP无法识别的细粒度属性冲突与常识谬误
    """
    
    def __init__(self):
        """
        初始化逻辑推理引擎
        
        架构说明:
          - 理想架构: Image -> VLM -> Caption -> LLM -> Conflict Check
          - Demo实现: Image -> (Query Excel Meta) -> Mock Caption -> Rule Check
        
        此实现采用 Demo实现 策略，以确保演示时的响应速度和绝对准确率。
        
        与 Channel 2 (CLIP) 的区分:
          - Ch2 (语义): 图和文主题是否一致 (Topic Alignment)
          - Ch3 (逻辑): 图和文在细节属性上是否矛盾 (Fact Verification)
          - Type 4 样本设计: 骗过 Ch2 (主题一致), 被 Ch3 抓获 (属性冲突)
        """
        print("[Ch3-Init] Initializing Logic Reasoning Engine (VLM-CoT)...")
        print("[Ch3-Init] Mode: Mock (Excel Meta as Oracle)")
        
        # =====================================================================
        # 时间关键词库 (Temporal Keywords)
        # 用于检测: 图(Day) vs 文(Night) 或 图(Night) vs 文(Day)
        # =====================================================================
        # 统一使用小写词库，后续匹配基于 lower 文本避免大小写问题
        self.night_keywords = [
            "深夜", "凌晨", "漆黑", "晚间", "通宵", "月色", "月光", "夜幕", "夜晚", "星空",
            "midnight", "night", "evening", "dark", "moon", "moonlight"
        ]
        self.day_keywords = [
            "阳光", "正午", "白天", "烈日", "上午", "下午", "中午", "清晨", "朝阳",
            "noon", "daylight", "morning", "daytime", "sunny", "afternoon"
        ]
        
        # =====================================================================
        # 天气关键词库 (Weather Keywords)
        # 用于检测: 图(Sunny) vs 文(Storm/Rain/Snow)
        # =====================================================================
        self.storm_keywords = [
            "暴雨", "洪水", "台风", "积水", "雷电", "狂风", "暴风", "大雨", "倾盆大雨", "风暴",
            "storm", "rain", "flood", "typhoon", "hurricane", "heavy rain", "rainstorm"
        ]
        self.snow_keywords = [
            "大雪", "暴雪", "寒冬", "冰雪", "雪花", "白雪皑皑", "鹅毛大雪",
            "snow", "blizzard", "snowstorm", "winter", "freezing", "frost"
        ]
        self.sunny_keywords = [
            "晴朗", "阳光明媚", "蓝天", "万里无云", "艳阳高照", "晴空万里",
            "sunny", "clear sky", "sunshine", "bright", "clear"
        ]
        
        # =====================================================================
        # 季节/温度关键词库 (Season/Temperature Keywords)
        # 用于检测: 图(夏天短袖) vs 文(大雪纷飞) 等常识冲突
        # =====================================================================
        self.winter_keywords = ["大雪", "寒冬", "冰雪", "snow", "winter", "freezing", "cold"]
        self.summer_keywords = ["炎热", "酷暑", "短袖", "summer", "hot", "scorching"]
        
        # =====================================================================
        # 数量/事实关键词库 (Quantity/Fact Keywords)
        # 用于检测: 图(空旷) vs 文(人山人海) 等事实冲突
        # =====================================================================
        self.crowded_keywords = [
            "人山人海", "座无虚席", "人满为患", "拥挤", "人潮涌动", "熙熙攘攘",
            "crowded", "packed", "full house"
        ]
        self.empty_keywords = [
            "空无一人", "空荡荡", "无人", "冷清", "空旷",
            "empty", "deserted", "nobody"
        ]

        # V2.0: 状态/OCR 反义词库
        self.state_conflicts = {
            "closed": ["敞开", "欢迎", "open", "welcome"],
            "sleeping": ["飞奔", "追逐", "running", "active"],
            "empty": ["丰盛", "满", "拥挤", "人山人海", "full", "crowded"],
            "red light": ["绿灯", "通行", "green light", "go"],
            "broken": ["全新", "完美", "无瑕", "brand new", "perfect"],
            "cracked": ["全新", "完美", "无瑕", "brand new", "perfect"],
            "sold out": ["充足", "现货", "available", "in stock"],
            "0-5": ["领先", "胜券在握", "winning", "leading"],
            "handshake": ["冲突", "打架", "破裂", "conflict", "fight"],
            "red card": ["继续", "无犯规", "合理", "continue", "no foul"]
        }

        # V2.0: Topic/双关检测关键词
        self.finance_kws = ["a股", "牛市", "熊市", "股市", "大盘", "指数", "黑天鹅事件"]
        self.crash_kws = ["暴跌", "崩盘", "价格", "泡沫破裂", "资产"]
        
        self.model_loaded = False  # 标记模型是否已加载

    def _vlm_captioning_mock(self, image_path, meta_data):
        """
        [模拟] VLM 的看图说话功能
        利用 Excel 里的 Meta 数据来'伪装'成 VLM 的视觉感知输出
        
        Args:
            image_path (str): 图片路径
            meta_data (dict): Excel 元数据
        
        Returns:
            dict: 结构化的视觉描述
        
        对应 Excel 字段:
            - Meta_Time: Day / Night
            - Meta_Weather: Sunny / Rain / Snow / Cloudy
            - Meta_Location: 地点名词 (Paris, Street...)
            - Meta_Fact: 事实状态 (Empty, Crowded...)
            - Meta_Object: 关键物体 (Car, Fire...)
        """
        # 从 Excel 元数据中获取真实信息，如果为空则默认为 Unknown
        time_info = str(meta_data.get('Meta_Time', 'Unknown')).strip()
        weather_info = str(meta_data.get('Meta_Weather', 'Unknown')).strip()
        location_info = str(meta_data.get('Meta_Location', 'Unknown')).strip()
        fact_info = str(meta_data.get('Meta_Fact', 'Unknown')).strip()
        object_info = str(meta_data.get('Meta_Object', 'Unknown')).strip()
        topic_info = str(meta_data.get('Meta_Topic', 'Unknown')).strip()
        
        # 构造结构化的视觉描述，模拟 VLM 的输出格式
        vlm_caption = {
            "Time": time_info,        # e.g., "Day", "Night"
            "Weather": weather_info,  # e.g., "Sunny", "Rain"
            "Location": location_info, # e.g., "Beach", "Street"
            "Fact": fact_info,        # e.g., "Noon", "Crowded"
            "Objects": object_info,   # e.g., "Eiffel Tower", "Car"
            "Topic": topic_info       # e.g., "Animal", "Sports"
        }
        return vlm_caption

    def _call_vlm_api(self, image_path, prompt):
        """
        [预留接口] 调用真实 VLM API
        
        TODO: 接入 Moondream / LLaVA 等 VLM 模型
        
        Args:
            image_path (str): 图片路径
            prompt (str): VLM 提示词
        
        Returns:
            str: VLM 生成的图片描述
        """
        # TODO: 实现真实 VLM 调用
        # from transformers import AutoModelForVision2Seq, AutoProcessor
        # processor = AutoProcessor.from_pretrained("vikhyatk/moondream2")
        # model = AutoModelForVision2Seq.from_pretrained("vikhyatk/moondream2")
        # image = Image.open(image_path)
        # inputs = processor(images=image, text=prompt, return_tensors="pt")
        # outputs = model.generate(**inputs)
        # caption = processor.decode(outputs[0], skip_special_tokens=True)
        # return caption
        pass

    def reasoning(self, image_path, text, meta_data):
        """
        执行逻辑推理
        
        Args:
            image_path (str): 图片路径 (e.g., "./data/images/real_noon.jpg")
            text (str): 新闻文本
            meta_data (dict): Excel 中的元数据 (Ground Truth / Oracle)
            
        Returns:
            is_conflict (bool): 是否冲突，对应 GT_Ch3_Logic
                                True = 有冲突(1), False = 无冲突(0)
            reason (str): 推理过程描述
        
        对应 Excel 字段:
            - 输入: Image_Path (C列), Text_Content (D列)
            - 验证: GT_Ch3_Logic (H列), 1=有冲突, 0=无冲突
            - 分类: Sample_Type (B列), Logic_Trap 类型需重点检测
        """
        file_name = os.path.basename(image_path)
        print(f"[Ch3-Analysis] Processing: {file_name}")
        
        # 1. [Vision Step] 视觉理解 (通过 Mock 获取)
        visual_facts = self._vlm_captioning_mock(image_path, meta_data)
        print(f"[Ch3-Analysis] VLM Observation: {visual_facts}")

        # 2. [Reasoning Step] 逻辑比对
        return self._mock_reasoning(visual_facts, text, image_path)

    def _mock_reasoning(self, visual_facts, text, image_path=""):
        """
        Mock 推理逻辑
        基于规则的逻辑冲突检测
        
        【与 Channel 2 的关键区分】
        Type 4 样本设计原则: "骗过 Ch2, 被 Ch3 抓获"
          - Ch2 (CLIP) 只看主题: 图是塔, 文说塔 -> 高分通过
          - Ch3 (Logic) 看细节: 图是巴黎塔, 文说东京塔 -> 冲突报警
        
        对应 Sample_Type = "Logic_Trap" 的检测:
          - ID=041: 时间冲突 (图白天, 文深夜)
          - ID=042: 天气冲突 (图晴天, 文暴雨)
        
        命名规范:
          - logic_001.jpg ~ logic_025.jpg (Logic_Trap 样本)
        
        检测规则 (按 Type 4 SOP):
          A. 时间陷阱 (Time Conflict) - 8张
          B. 天气陷阱 (Weather Conflict) - 7张
          C. 地标/实体错位 (Entity Mismatch) - 5张
          D. 数量/事实冲突 (Fact Conflict) - 5张
          E. 显性触发词 (Manual Trigger)
        """
        conflict_detected = False
        reason = "[CONSISTENT] Visual evidence aligns with text description"

        img_time = visual_facts.get("Time", "Unknown")
        img_weather = visual_facts.get("Weather", "Unknown")
        img_location = visual_facts.get("Location", "Unknown")
        img_fact = visual_facts.get("Fact", "Unknown")
        img_objects = visual_facts.get("Objects", "Unknown")
        img_topic = visual_facts.get("Topic", "Unknown")

        # 归一化，确保大小写/空格不影响匹配
        text_norm = str(text).lower()
        img_time_norm = str(img_time).lower()
        img_weather_norm = str(img_weather).lower()
        img_location_norm = str(img_location).lower()
        img_fact_norm = str(img_fact).lower()
        img_objects_norm = str(img_objects).lower()
        img_topic_norm = str(img_topic).lower()
        
        # 获取文件名用于额外匹配
        file_name = os.path.basename(image_path).lower() if image_path else ""

        # =================================================================
        # Rule A: 时间陷阱 (Time Conflict) - 8张
        # 强烈的视觉光影冲突
        # logic_001: 正午故宫 + "深夜月光"
        # logic_002: 纽约夜景 + "今天上午"
        # =================================================================
        if img_time_norm == "day":
            matched_kw = next((kw for kw in self.night_keywords if kw in text_norm), None)
            if matched_kw:
                conflict_detected = True
                reason = f"[CONFLICT] Temporal: Image shows '{img_time}' (day) vs Text mentions '{matched_kw}'"
        elif img_time_norm == "night":
            matched_kw = next((kw for kw in self.day_keywords if kw in text_norm), None)
            if matched_kw:
                conflict_detected = True
                reason = f"[CONFLICT] Temporal: Image shows '{img_time}' (night) vs Text mentions '{matched_kw}'"

        # =================================================================
        # Rule B: 天气陷阱 (Weather Conflict) - 7张
        # 氛围与环境冲突
        # logic_003: 阳光海滩 + "台风狂风暴雨"
        # logic_004: 漫天大雪 + "炎热夏天"
        # =================================================================
        if not conflict_detected:
            sunny_scene = any(kw in img_weather_norm for kw in ["sunny", "clear", "晴"])
            storm_hit = next((kw for kw in self.storm_keywords if kw.lower() in text_norm), None)
            snow_hit = next((kw for kw in self.snow_keywords if kw.lower() in text_norm), None)
            summer_hit = next((kw for kw in self.summer_keywords if kw.lower() in text_norm), None)
            sunny_hit = next((kw for kw in self.sunny_keywords if kw.lower() in text_norm), None)

            if sunny_scene and storm_hit:
                conflict_detected = True
                reason = f"[CONFLICT] Weather: Image '{img_weather}' (sunny/clear) vs Text '{storm_hit}'"
            if not conflict_detected and sunny_scene and snow_hit:
                conflict_detected = True
                reason = f"[CONFLICT] Weather: Image '{img_weather}' (sunny) vs Text '{snow_hit}'"

            snow_scene = any(kw in img_weather_norm for kw in ["snow", "雪", "winter"])
            if not conflict_detected and snow_scene and summer_hit:
                conflict_detected = True
                reason = f"[CONFLICT] Weather: Image '{img_weather}' (snow) vs Text '{summer_hit}'"

            rain_scene = any(kw in img_weather_norm for kw in ["rain", "雨", "storm"])
            if not conflict_detected and rain_scene and sunny_hit:
                conflict_detected = True
                reason = f"[CONFLICT] Weather: Image '{img_weather}' (rain) vs Text '{sunny_hit}'"

        # =================================================================
        # Rule C: 地标/实体错位 (Entity Mismatch) - 5张
        # "看起来很像，其实不是"
        # logic_005: 埃菲尔铁塔 + "东京塔"
        # logic_006: 悉尼歌剧院 + "北京"
        # =================================================================
        if not conflict_detected:
            # C1: 巴黎地标 vs 其他城市
            paris_landmarks = ["eiffel tower", "埃菲尔铁塔", "巴黎", "paris"]
            wrong_cities_for_paris = ["tokyo", "东京", "london", "伦敦", "beijing", "北京", "new york", "纽约"]

            if any(lm in img_objects_norm or lm in img_location_norm for lm in paris_landmarks):
                matched_city = next((loc for loc in wrong_cities_for_paris if loc in text_norm), None)
                if matched_city:
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows Paris/Eiffel Tower vs Text '{matched_city}'"
            
            # C2: 东京地标 vs 其他城市
            tokyo_landmarks = ["tokyo tower", "东京塔", "tokyo", "东京"]
            wrong_cities_for_tokyo = ["paris", "巴黎", "eiffel", "埃菲尔", "london", "伦敦", "shanghai", "上海"]

            if not conflict_detected and any(lm in img_objects_norm or lm in img_location_norm for lm in tokyo_landmarks):
                matched_city = next((loc for loc in wrong_cities_for_tokyo if loc in text_norm), None)
                if matched_city:
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows Tokyo vs Text '{matched_city}'"
            
            # C3: 上海地标 vs 其他城市
            shanghai_landmarks = ["东方明珠", "陆家嘴", "shanghai", "上海"]
            wrong_cities_for_shanghai = ["tokyo", "东京", "beijing", "北京", "hong kong", "香港"]

            if not conflict_detected and any(lm in img_objects_norm or lm in img_location_norm for lm in shanghai_landmarks):
                matched_city = next((loc for loc in wrong_cities_for_shanghai if loc in text_norm), None)
                if matched_city:
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows Shanghai vs Text '{matched_city}'"
            
            # C4: 悉尼地标 vs 其他城市
            sydney_landmarks = ["sydney opera house", "悉尼歌剧院", "sydney", "悉尼"]
            wrong_cities_for_sydney = ["beijing", "北京", "tokyo", "东京", "london", "伦敦"]

            if not conflict_detected and any(lm in img_objects_norm or lm in img_location_norm for lm in sydney_landmarks):
                matched_city = next((loc for loc in wrong_cities_for_sydney if loc in text_norm), None)
                if matched_city:
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows Sydney vs Text '{matched_city}'"
            
            # C5: 伦敦塔桥 vs 金门大桥
            london_bridge = ["london bridge", "tower bridge", "伦敦塔桥", "伦敦桥"]
            golden_gate = ["golden gate", "金门大桥", "san francisco", "旧金山"]

            if not conflict_detected and any(lm in img_objects_norm or lm in img_location_norm for lm in london_bridge):
                matched_city = next((loc for loc in golden_gate if loc in text_norm), None)
                if matched_city:
                    conflict_detected = True
                    reason = f"[CONFLICT] Geolocation: Image shows London Bridge vs Text '{matched_city}'"

        # =================================================================
        # Rule D: 数量/事实冲突 (Fact Conflict) - 5张
        # 明显的视觉事实矛盾
        # 图: 空荡荡的会议室 + 文: "座无虚席，人山人海"
        # =================================================================
        if not conflict_detected:
            # D1: 空旷场景 vs 拥挤描述
            empty_scene = any(kw.lower() in img_fact_norm for kw in ["empty", "空", "no people", "无人", "deserted"])
            crowded_hit = next((kw for kw in self.crowded_keywords if kw.lower() in text_norm), None)
            if empty_scene and crowded_hit:
                conflict_detected = True
                reason = f"[CONFLICT] Fact: Image '{img_fact}' (empty) vs Text '{crowded_hit}'"

            crowded_scene = any(kw.lower() in img_fact_norm for kw in ["crowded", "拥挤", "人多", "many people"])
            empty_hit = next((kw for kw in self.empty_keywords if kw.lower() in text_norm), None)
            if not conflict_detected and crowded_scene and empty_hit:
                conflict_detected = True
                reason = f"[CONFLICT] Fact: Image '{img_fact}' (crowded) vs Text '{empty_hit}'"

        # =================================================================
        # Rule E: 显性逻辑谬误触发词 (Manual Trigger for Demo)
        # 在演示时，如果文本中包含特定词，强制触发报警
        # =================================================================
        if not conflict_detected:
            trigger_keywords = ["逻辑错误", "明显矛盾", "logic_trap", "conflict_test"]
            matched_kw = next((kw for kw in trigger_keywords if kw in text_norm), None)
            if matched_kw:
                conflict_detected = True
                reason = f"[CONFLICT] Manual Trigger: '{matched_kw}' detected"

        # =================================================================
        # Rule G: 状态/OCR 冲突 (V2.0 新增)
        # =================================================================
        if not conflict_detected:
            for state_key, anti_keywords in self.state_conflicts.items():
                if state_key in img_fact_norm or state_key in img_objects_norm:
                    matched_anti = next((ak for ak in anti_keywords if ak.lower() in text_norm), None)
                    if matched_anti:
                        conflict_detected = True
                        reason = f"[CONFLICT] State/OCR: Image '{state_key}' vs Text '{matched_anti}'"
                        break

        # =================================================================
        # Rule H: Topic / 双关冲突 (V2.0 新增)
        # =================================================================
        if not conflict_detected:
            if img_topic_norm in ["animal", "living animal", "动物", "生物"]:
                matched_finance = next((kw for kw in self.finance_kws if kw in text_norm), None)
                if matched_finance:
                    conflict_detected = True
                    reason = f"[CONFLICT] Polysemy: Image real animal vs Text finance '{matched_finance}'"
            elif img_topic_norm in ["sports", "object", "soap bubble", "体育", "物体", "泡泡"]:
                matched_crash = next((kw for kw in self.crash_kws if kw in text_norm), None)
                if matched_crash:
                    conflict_detected = True
                    reason = f"[CONFLICT] Polysemy: Image physical object vs Text economic '{matched_crash}'"
        
        # =================================================================
        # Rule F: 文件名前缀检测 (Naming Convention)
        # 如果文件名以 logic_ 开头，增加检测敏感度
        # =================================================================
        if not conflict_detected and file_name.startswith("logic_"):
            # logic_ 前缀的文件应该触发冲突检测
            # 这里作为兜底检测，实际应该通过上述规则捕获
            print(f"[Ch3-Warning] File '{file_name}' has logic_ prefix but no conflict detected. Check Excel Meta fields.")

        # -----------------------------------------------------------------
        # 返回结果
        # -----------------------------------------------------------------
        if conflict_detected:
            print(f"[Ch3-Result] Status=Conflict, Reason={reason}")
        else:
            print(f"[Ch3-Result] Status=Consistent")
        
        return conflict_detected, reason


# ============================================================================
# 单例模式导出
# ============================================================================
reasoner = LogicReasoner()


def check_logic(image_path, text, meta_data):
    """
    外部调用接口 (标准函数)
    供 main.py 或其他模块调用
    
    Args:
        image_path (str): 图片路径
        text (str): 新闻文本
        meta_data (dict): Excel 元数据字典，需包含:
            - Meta_Time: Day / Night
            - Meta_Weather: Sunny / Rain / Snow / Cloudy
            - Meta_Location: 地点名词
            - Meta_Fact: 事实状态
            - Meta_Object: 关键物体
    
    Returns:
        tuple: (is_conflict, reason)
            - is_conflict (bool): True=有冲突, False=无冲突
            - reason (str): 推理证据
    
    注意:
        - 返回的 is_conflict 直接对应 GT_Ch3_Logic
        - True = 1 (有冲突), False = 0 (无冲突)
    
    使用示例:
        from channel_3_logic_rules.reasoner import check_logic
        meta = {"Meta_Time": "Day", "Meta_Weather": "Sunny", ...}
        is_conflict, reason = check_logic("data/images/real_noon.jpg", 
                                          "深夜的街道格外宁静", meta)
        P3 = 0.95 if is_conflict else 0.05
        print(f"Conflict={is_conflict}, P3={P3}, Reason: {reason}")
    """
    return reasoner.reasoning(image_path, text, meta_data)


def check_logic_pipeline(image_path, text, meta_data):
    """
    Pipeline 接口 (别名)
    兼容不同的调用方式
    """
    return reasoner.reasoning(image_path, text, meta_data)


def run_ch3_csv(csv_path="channel_3_logic_rules/ch3_dataset.csv", output_path="channel_3_logic_rules/ch3_results.csv", image_base_dir=None):
    """
    读取仅包含通道三字段的 CSV 并批量推理。

    约定：
      - csv_path 默认放在与 reasoner.py 同目录。
      - Image_Path 如果是相对路径且传入 image_base_dir，则前缀拼接。
      - Mock 模式下不依赖真实图片内容，image_path 仅用于日志展示。
    """
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        image_path = str(row.get("Image_Path", "")).strip()
        if image_base_dir and image_path and not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        meta = {
            "Meta_Time": row.get("Meta_Time", ""),
            "Meta_Weather": row.get("Meta_Weather", ""),
            "Meta_Location": row.get("Meta_Location", ""),
            "Meta_Fact": row.get("Meta_Fact", ""),
            "Meta_Object": row.get("Meta_Object", ""),
            "Meta_Topic": row.get("Meta_Topic", ""),
        }
        is_conflict, reason = check_logic(image_path, row.get("Text_Content", ""), meta)
        results.append({
            "ID": row.get("ID", ""),
            "Image_Path": row.get("Image_Path", ""),
            "Text_Content": row.get("Text_Content", ""),
            "Pred_Ch3_Conflict": is_conflict,
            "Reason": reason,
        })

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[Ch3-Batch] Saved results -> {output_path}")


def run_ch3_excel(excel_path="channel_3_logic_rules/ch3_dataset.xlsx", sheet_name=0, output_path="channel_3_logic_rules/ch3_results.xlsx", image_base_dir=None):
    """
    读取 xlsx 并批量推理，方便直接放同目录的 Excel。

    Args:
        excel_path: Excel 路径（默认与 reasoner.py 同目录）。
        sheet_name: 读取的 sheet 名或索引，默认第一个 sheet。
        output_path: 输出结果 xlsx 路径。
        image_base_dir: 可选，为相对 Image_Path 提供前缀。
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    results = []
    for _, row in df.iterrows():
        image_path = str(row.get("Image_Path", "")).strip()
        if image_base_dir and image_path and not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        meta = {
            "Meta_Time": row.get("Meta_Time", ""),
            "Meta_Weather": row.get("Meta_Weather", ""),
            "Meta_Location": row.get("Meta_Location", ""),
            "Meta_Fact": row.get("Meta_Fact", ""),
            "Meta_Object": row.get("Meta_Object", ""),
            "Meta_Topic": row.get("Meta_Topic", ""),
        }
        is_conflict, reason = check_logic(image_path, row.get("Text_Content", ""), meta)
        results.append({
            "ID": row.get("ID", ""),
            "Image_Path": row.get("Image_Path", ""),
            "Text_Content": row.get("Text_Content", ""),
            "Pred_Ch3_Conflict": is_conflict,
            "Reason": reason,
        })

    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"[Ch3-Batch] Saved results -> {output_path}")


def run_evaluation(dataset_path="channel_3_logic_rules/Yuanjing_Data_Standard_Channel_3.xlsx", sheet_name=0):
    """
    运行评估测试并打印详细报告 (From User Request)
    """
    print(f"🚀 Running Channel 3 Evaluation...")
    print(f"📂 Loading dataset: {dataset_path}")
    
    try:
        # 兼容 CSV 和 Excel
        if dataset_path.lower().endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {dataset_path}")
        return
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return

    total = 0
    correct = 0
    
    # 打印表头
    print("-" * 120)
    print(f"{'ID':<6} | {'Visual (Meta)':<30} | {'Text Keyword':<20} | {'GT':<3} | {'Pred':<4} | {'Result'}")
    print("-" * 120)

    for idx, row in df.iterrows():
        # 构造 Meta
        meta = {
            "Meta_Time": row.get('Meta_Time', ''),
            "Meta_Weather": row.get('Meta_Weather', ''),
            "Meta_Location": row.get('Meta_Location', ''),
            "Meta_Fact": row.get('Meta_Fact', ''),
            "Meta_Object": row.get('Meta_Object', ''),
            "Meta_Topic": row.get('Meta_Topic', '')
        }
        
        text = str(row.get('Text_Content', ''))
        
        try:
            gt = int(row['GT_Ch3_Logic'])
        except (ValueError, KeyError):
            gt = -1
        
        # 运行推理
        image_path = str(row.get('Image_Path', ''))
        is_conflict, reason = check_logic(image_path, text, meta)
        pred = 1 if is_conflict else 0
        
        # 统计
        res_icon = "❓"
        if gt != -1:
            total += 1
            if pred == gt:
                correct += 1
                res_icon = "✅"
            else:
                res_icon = "❌"
        
        # 提取关键视觉信息用于展示
        meta_values = [str(v) for k, v in meta.items() if v and str(v).lower() != 'nan' and str(v).lower() != 'unknown']
        visual_cue = "/".join(meta_values)
        if len(visual_cue) > 30: visual_cue = visual_cue[:27] + "..."
        
        text_preview = text.replace('\\n', ' ')
        if len(text_preview) > 20: text_preview = text_preview[:17] + "..."
        
        print(f"{str(row.get('ID', idx)):<6} | {visual_cue:<30} | {text_preview:<20} | {gt:<3} | {pred:<4} | {res_icon}")
        
        if gt != -1 and pred != gt:
             print(f"      >>> Engine Reason: {reason}")
             print(f"      >>> Text Full: {text}")

    print("-" * 120)
    if total > 0:
        print(f"📊 Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    else:
        print("⚠️ No valid Ground Truth found.")


if __name__ == "__main__":
    # 自动探测文件路径
    possible_paths = [
        "channel_3_logic_rules/Yuanjing_Data_Standard_Channel_3.xlsx",
        "Yuanjing_Data_Standard_Channel_3.xlsx"
    ]
    
    selected_path = None
    for p in possible_paths:
        if os.path.exists(p):
            selected_path = p
            break
            
    if selected_path:
        run_evaluation(selected_path)
    else:
        print("⚠️ Default dataset not found in common locations.")
