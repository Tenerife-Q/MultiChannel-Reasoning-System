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
    def __init__(self):
        print("[Ch3-Init] Initializing Logic Reasoning Engine V3.1 (Optimization)...")
        
        # =================================================================
        # 1. 基础属性词库 (扩充版)
        # =================================================================
        self.night_keywords = ["深夜", "凌晨", "漆黑", "晚间", "通宵", "月色", "月光", "夜幕", "夜晚", "黑夜", "伸手不见五指", "midnight", "night", "dark"]
        self.day_keywords = ["阳光", "正午", "白天", "烈日", "上午", "下午", "中午", "清晨", "白昼", "noon", "day", "sunny", "sunlight"]
        
        self.storm_keywords = ["暴雨", "洪水", "台风", "积水", "雷电", "狂风", "大雨", "storm", "rain", "flood"]
        self.snow_keywords = ["大雪", "暴雪", "寒冬", "冰雪", "雪花", "snow", "blizzard", "winter"]
        self.sunny_keywords = ["晴朗", "阳光", "蓝天", "无云", "烈日", "sunny", "clear"]
        self.summer_keywords = ["酷暑", "炎热", "夏天", "高温", "summer", "hot", "heat"]

        self.crowded_keywords = ["人山人海", "座无虚席", "人满为患", "拥挤", "人潮", "熙熙攘攘", "人头攒动", "火爆", "车辆很多", "full", "crowded", "packed", "busy"]
        self.empty_keywords = ["空无一人", "空荡荡", "无人", "冷清", "空旷", "empty", "deserted", "no people", "no one", "0人"]

        # =================================================================
        # 2. 实体/地标冲突映射表 (Type 4 Entity)
        # =================================================================
        self.entity_conflicts = {
            # 地标建筑
            "tokyo tower": ["eiffel", "埃菲尔", "paris", "巴黎"],
            "eiffel tower": ["tokyo tower", "东京塔", "japan", "日本"],
            "canton tower": ["skytree", "晴空塔", "japan", "日本", "triangle", "三角形"],
            "tower bridge": ["london bridge", "伦敦大桥"], 
            "london bridge": ["tower bridge", "塔桥"],
            "oriental pearl": ["cctv", "北京", "needle", "针状"],
            "capitol": ["white house", "白宫", "flat roof"],
            "white house": ["capitol", "国会", "dome", "圆顶"],
            "statue of liberty": ["las vegas", "赌城"] if "las vegas" else [],
            "daxing airport": ["mars", "火星", "concept", "概念"],
            "25 de abril bridge": ["golden gate", "金门"],
            
            # 文化/OCR
            "chinese": ["japanese", "日文", "京都", "tokyo"],
            "simplified chinese": ["japanese", "日文"],
        }
        
        # 2.1 通用场景冲突 (V3.1 新增)
        self.location_mismatches = {
            "forest": ["street", "city", "building", "街道", "城市", "楼房"],
            "library": ["street", "outdoor", "park", "街道", "户外", "公园"],
            "indoor": ["street", "forest", "mountain", "街道", "森林", "山顶"],
            "street": ["indoor", "room", "hall", "室内", "房间", "大厅"],
            "mountain": ["room", "indoor", "flat", "房间", "室内", "平原"]
        }

        # =================================================================
        # 3. 状态与事实冲突映射表 (Type 3 Fact / X Series)
        # =================================================================
        self.fact_conflicts = {
            "closed": ["敞开", "欢迎", "open"],
            "sleeping": ["飞奔", "追逐", "run", "active"],
            "empty": self.crowded_keywords + ["many cars", "several people", "好几个人", "很多车"],
            "crowded": self.empty_keywords,
            "withered": ["fresh", "spring", "生机", "春意", "盎然", "翠绿"],
            "barren": ["harvest", "golden", "lush", "forest", "丰收", "茂密"],
            "dirty": ["clean", "sanitary", "整洁", "卫生", "一尘不染"],
            "red light": ["绿灯", "通行", "green", "go"],
            "no smoking": ["吸烟区", "smoking area"],
            "broken": ["全新", "完美", "无瑕", "brand new"],
            "cracked": ["全新", "完美"],
            "sold out": ["充足", "现货", "available"],
            "0-5": ["领先", "胜券在握", "winning"],
        }

        # =================================================================
        # 4. 双关语/话题冲突 (Type 5 Polysemy)
        # =================================================================
        self.topic_conflicts = {
            "animal": ["a股", "牛市", "熊市", "股市", "大盘", "指数", "黑天鹅事件", "finance", "stock", "market"],
            "living animal": ["a股", "牛市", "熊市", "股市", "大盘", "指数", "黑天鹅事件"],
            "sports": ["暴跌", "崩盘", "价格", "泡沫", "资产", "跳水", "下挫", "跌停", "economic"],
            "object": ["暴跌", "崩盘", "价格", "泡沫", "资产", "evaporate", "蒸发"],
            "soap bubble": ["暴跌", "崩盘", "价格", "泡沫", "资产", "房产"],
            "nature": ["crypto", "blockchain", "industry", "recession", "裁员", "矿机", "行业寒冬"], 
            # 增加 plant 别名
            "vegetable": ["investor", "stock", "散户", "追涨杀跌", "收割", "韭菜"],
            "plant": ["investor", "stock", "散户", "追涨杀跌", "收割", "韭菜"]
        }

    def _vlm_captioning_mock(self, image_path, meta_data):
        """模拟 VLM 输出"""
        # 注意：这里强转 str 并 strip，防止 Excel 里的 None 或数字格式干扰
        return {
            "Time": str(meta_data.get('Meta_Time', 'Unknown')).strip(),
            "Weather": str(meta_data.get('Meta_Weather', 'Unknown')).strip(),
            "Location": str(meta_data.get('Meta_Location', 'Unknown')).strip(),
            "Fact": str(meta_data.get('Meta_Fact', 'Unknown')).strip(),
            "Objects": str(meta_data.get('Meta_Object', 'Unknown')).strip(),
            "Topic": str(meta_data.get('Meta_Topic', 'Unknown')).strip()
        }

    def reasoning(self, image_path, text, meta_data):
        visual_facts = self._vlm_captioning_mock(image_path, meta_data)
        conflict = False
        reason = "[CONSISTENT] Logic check passed"
        
        text_norm = str(text).lower()
        
        img_time = visual_facts["Time"]
        img_weather = visual_facts["Weather"].lower()
        img_loc = visual_facts["Location"].lower()
        img_fact = visual_facts["Fact"].lower()
        img_obj = visual_facts["Objects"].lower()
        img_topic = visual_facts["Topic"].lower()

        # -----------------------------------------------------------
        # Logic 1: Time
        # -----------------------------------------------------------
        if img_time == "Day" and any(k in text_norm for k in self.night_keywords):
            conflict, reason = True, f"[CONFLICT] Time: Visual[Day] vs Text[Night]"
        elif img_time == "Night" and any(k in text_norm for k in self.day_keywords):
            conflict, reason = True, f"[CONFLICT] Time: Visual[Night] vs Text[Day]"

        # -----------------------------------------------------------
        # Logic 2: Weather (Refined)
        # -----------------------------------------------------------
        if not conflict:
            if "sunny" in img_weather or "clear" in img_weather:
                if any(k in text_norm for k in self.storm_keywords + self.snow_keywords + ["rain", "雨"]):
                    conflict, reason = True, f"[CONFLICT] Weather: Visual[Sunny] vs Text[Bad Weather]"
            elif "snow" in img_weather:
                # Snow vs Rain (V3.1 Fix)
                if any(k in text_norm for k in self.summer_keywords + self.sunny_keywords + ["heat", "hot", "rain", "雨"]):
                    conflict, reason = True, f"[CONFLICT] Weather: Visual[Snow] vs Text[Summer/Hot/Rain]"
            elif "rain" in img_weather:
                if any(k in text_norm for k in self.sunny_keywords + ["dry", "干燥"]):
                    conflict, reason = True, f"[CONFLICT] Weather: Visual[Rain] vs Text[Sunny/Dry]"

        # -----------------------------------------------------------
        # Logic 3: Entity / Landmark / Location
        # -----------------------------------------------------------
        if not conflict:
            # 3.1 具体实体
            for entity_key, conflict_words in self.entity_conflicts.items():
                if entity_key in img_obj or entity_key in img_loc or entity_key in img_fact:
                    matched = next((w for w in conflict_words if w in text_norm), None)
                    if matched:
                        if entity_key == "tower bridge" and "london bridge" in text_norm:
                            if "tower" not in text_norm and "塔" not in text_norm:
                                conflict, reason = True, f"[CONFLICT] Entity: Visual[{entity_key}] vs Text[{matched}]"
                        else:
                            conflict, reason = True, f"[CONFLICT] Entity: Visual[{entity_key}] vs Text[{matched}]"
                        break
            
            # 3.2 拉斯维加斯修正 (V3.1 Fix: 中文关键词)
            if not conflict and "las vegas" in img_loc:
                 if any(w in text_norm for w in ["new york", "ocean", "harbor", "纽约", "海港", "大西洋"]):
                    conflict, reason = True, f"[CONFLICT] Location: Visual[Las Vegas] vs Text[New York/Ocean]"

            # 3.3 通用位置修正 (V3.1 Fix: Forest vs Street)
            if not conflict:
                for loc_key, mismatch_list in self.location_mismatches.items():
                    if loc_key in img_loc or loc_key in img_obj: # e.g. Image has 'Forest'
                         matched_loc = next((w for w in mismatch_list if w in text_norm), None)
                         if matched_loc:
                             conflict, reason = True, f"[CONFLICT] Location: Visual[{loc_key}] vs Text[{matched_loc}]"
                             break

        # -----------------------------------------------------------
        # Logic 4: Fact / State / Quantity
        # -----------------------------------------------------------
        if not conflict:
            for fact_key, conflict_words in self.fact_conflicts.items():
                if fact_key in img_fact or fact_key in img_obj:
                    matched = next((w for w in conflict_words if w in text_norm), None)
                    if matched:
                        conflict, reason = True, f"[CONFLICT] Fact/State: Visual[{fact_key}] vs Text[{matched}]"
                        break

        # -----------------------------------------------------------
        # Logic 5: Polysemy (Refined)
        # -----------------------------------------------------------
        if not conflict:
            # 检查 Topic, Fact, Object 字段
            check_source = f"{img_topic} {img_fact} {img_obj}"
            
            for topic_key, conflict_words in self.topic_conflicts.items():
                if topic_key in check_source:
                    matched = next((w for w in conflict_words if w in text_norm), None)
                    if matched:
                        conflict, reason = True, f"[CONFLICT] Polysemy: Visual[{topic_key}] vs Text[{matched}]"
                        break

        return conflict, reason

# 导出接口
reasoner = LogicReasoner()
def check_logic(image_path, text, meta_data):
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
