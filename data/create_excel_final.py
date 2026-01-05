"""
================================================================================
YuanJing AI 数据标准生成脚本 (Final Ver. for Probability Fusion)
文件名: create_excel_final.py
输出: Yuanjing_Data_Standard_Final.xlsx

【设计哲学】
为了支持最终的 "多通道概率级联融合 (Multi-Channel Cascade Fusion)" 算法，
本表格的 Ground Truth (GT) 字段全部标准化为 "Risk Label" (风险标签)。
即：1 代表存在风险(Fake/Tampered/Conflict)，0 代表安全(Real/Consistent)。
这样设计可以极大简化后续的数学统计逻辑。

================================================================================
【字段详细解释 (16列)】
================================================================================

A. 索引与分类 (用于生成分层报告)
  - ID: 唯一编号
  - Sample_Type: 核心分类 [Real, Tamper_PS, Tamper_AIGC, Mismatch, Logic_Trap]
                 用于最后统计：系统在某一类的召回率是多少？

B. 输入数据
  - Image_Path: 图片路径
  - Text_Content: 文本内容

C. 概率验证标准答案 (Ground Truth for Math Stats)
   全部采用 1=Risk(Fake), 0=Safe(Real) 的逻辑
  - GT_Final_Label: 最终判定 1=Fake, 0=Real
  - GT_Ch1_Tamper: 通道1真值 1=有篡改, 0=无篡改 (验证 MVSS-Net 的 P1 准确率)
  - GT_Ch2_Mismatch: 通道2真值 1=图文不符, 0=一致 (直接计算 P2 命中率)
  - GT_Ch3_Logic: 通道3真值 1=有逻辑冲突, 0=无冲突 (验证 VLM-Reasoning 的 P3)

D. 辅助统计信息
  - Tamper_Method: 篡改手段备注 (e.g., "MagicEraser", "Photoshop_Clone")

E. 逻辑推理元数据 (Channel 3 Mock Data)
   reasoner.py 读取这些字段作为"上帝视角"
  - Meta_Time: Day / Night
  - Meta_Weather: Sunny / Rain / Snow / Cloudy
  - Meta_Location: 地点名词 (Paris, Street...)
  - Meta_Fact: 事实状态 (Empty, Crowded...)
  - Meta_Object: 关键物体 (Car, Fire...)

F. 人工备注
  - Source_Note: 详细描述，答辩备忘用

================================================================================
【概率融合公式参考】
================================================================================
最终虚假概率 P_final = 1 - (1-P1)*(1-P2)*(1-P3)
其中:
  P1 = 篡改检测概率 (对应 GT_Ch1_Tamper)
  P2 = 图文不符概率 (对应 GT_Ch2_Mismatch)
  P3 = 逻辑冲突概率 (对应 GT_Ch3_Logic)

================================================================================
"""

import pandas as pd
import os

# ============================================================================
# 1. 字段定义 (16列) - 完美契合概率统计模型
# ============================================================================
columns_order = [
    # --- A. 索引与分类 (用于生成分层报告) ---
    "ID",               # 唯一编号
    "Sample_Type",      # 核心分类: [Real, Tamper_PS, Tamper_AIGC, Mismatch, Logic_Trap]
                        # ^ 用于最后统计：系统在 "Logic_Trap" 这一类的召回率是多少？

    # --- B. 输入数据 ---
    "Image_Path",       # 图片路径
    "Text_Content",     # 文本内容

    # --- C. 概率验证标准答案 (Ground Truth for Math Stats) ---
    # 全部采用 1=Risk(Fake), 0=Safe(Real) 的逻辑
    "GT_Final_Label",   # 最终判定: 1=Fake, 0=Real
    
    "GT_Ch1_Tamper",    # 通道1真值: 1=有篡改, 0=无篡改
                        # (用于验证 MVSS-Net 的 P1 准确率)
                        
    "GT_Ch2_Mismatch",  # 通道2真值: 1=图文不符, 0=一致
                        # (注意：这里存的是"不符"标签，方便直接计算 P2 命中率)
                        
    "GT_Ch3_Logic",     # 通道3真值: 1=有逻辑冲突, 0=无冲突
                        # (用于验证 VLM-Reasoning 的 P3)

    # --- D. 辅助统计信息 ---
    "Tamper_Method",    # 篡改手段备注 (e.g., "MagicEraser", "Photoshop_Clone")
    
    # --- E. 逻辑推理元数据 (Channel 3 Mock Data) ---
    # 代码 reasoner.py 读取这些字段作为"上帝视角"
    "Meta_Time",        # Day / Night
    "Meta_Weather",     # Sunny / Rain / Snow / Cloudy
    "Meta_Location",    # 地点名词 (Paris, Street...)
    "Meta_Fact",        # 事实状态 (Empty, Crowded...)
    "Meta_Object",      # 关键物体 (Car, Fire...)
    
    # --- F. 人工备注 ---
    "Source_Note"       # 详细描述，答辩备忘用
]

# ============================================================================
# 2. 构造样本的标准结构 (示例数据)
# ============================================================================
data = [
    # ------------------------------------------------------------------------
    # Group 1: 真实基准 (Type 1) - Real样本
    # ------------------------------------------------------------------------
    {
        "ID": "001",
        "Sample_Type": "Real",
        "Image_Path": "data/images/real_001.jpg",
        "Text_Content": "阳光明媚的陆家嘴金融中心。",
        # GT 全部为 0 (低风险)
        "GT_Final_Label": 0,
        "GT_Ch1_Tamper": 0,
        "GT_Ch2_Mismatch": 0,
        "GT_Ch3_Logic": 0,
        "Tamper_Method": "None",
        # Meta 如实填写
        "Meta_Time": "Day", "Meta_Weather": "Sunny", "Meta_Location": "Shanghai",
        "Meta_Fact": "Skyscrapers", "Meta_Object": "Building",
        "Source_Note": "完全真实样本"
    },

    # ------------------------------------------------------------------------
    # Group 2: 物理篡改 - AIGC篡改样本
    # ------------------------------------------------------------------------
    # Case A: AIGC 消除 (考验 Ch1)
    {
        "ID": "011",
        "Sample_Type": "Tamper_AIGC",
        "Image_Path": "data/images/aigc_eraser_001.jpg",
        "Text_Content": "街道非常干净，没有垃圾。",  # 配合假图说话，语义无冲突
        # GT: Ch1=1, Ch2=0 (匹配), Ch3=0
        "GT_Final_Label": 1,
        "GT_Ch1_Tamper": 1, 
        "GT_Ch2_Mismatch": 0, 
        "GT_Ch3_Logic": 0,
        "Tamper_Method": "MagicEraser",
        "Meta_Time": "Day", "Meta_Weather": "Cloudy", "Meta_Location": "Street",
        "Meta_Fact": "Clean", "Meta_Object": "Road",
        "Source_Note": "AIGC消除垃圾桶，测试噪声检测"
    },
    
    # ------------------------------------------------------------------------
    # Group 3: 物理篡改 - PS篡改样本
    # ------------------------------------------------------------------------
    # Case B: PS 拼接 (考验 Ch1)
    {
        "ID": "021",
        "Sample_Type": "Tamper_PS",
        "Image_Path": "data/images/ps_splicing_001.jpg",
        "Text_Content": "三架飞机编队飞行。",
        "GT_Final_Label": 1,
        "GT_Ch1_Tamper": 1, 
        "GT_Ch2_Mismatch": 0, 
        "GT_Ch3_Logic": 0,
        "Tamper_Method": "Photoshop_Splicing",
        "Meta_Time": "Day", "Meta_Weather": "Sunny", "Meta_Location": "Sky",
        "Meta_Fact": "3 Planes", "Meta_Object": "Plane",
        "Source_Note": "PS拼接飞机，测试边缘检测"
    },

    # ------------------------------------------------------------------------
    # Group 4: 语义不符 (移花接木) - Mismatch样本
    # ------------------------------------------------------------------------
    {
        "ID": "031",
        "Sample_Type": "Mismatch",
        "Image_Path": "data/images/real_fire.jpg",
        "Text_Content": "商场举办盛大促销活动。",
        # GT: Ch2=1 (Mismatch), Ch1=0
        "GT_Final_Label": 1,
        "GT_Ch1_Tamper": 0, 
        "GT_Ch2_Mismatch": 1,  # 1代表不匹配/高风险
        "GT_Ch3_Logic": 0,     # 语义都崩了，逻辑层通常视为 N/A 或 0
        "Tamper_Method": "None",
        "Meta_Time": "Day", "Meta_Weather": "Sunny", "Meta_Location": "Forest",
        "Meta_Fact": "Fire", "Meta_Object": "Tree",
        "Source_Note": "图文完全无关"
    },

    # ------------------------------------------------------------------------
    # Group 5: 逻辑陷阱 (高阶推理) - Logic_Trap样本
    # ------------------------------------------------------------------------
    # Case A: 时间冲突 (Time)
    {
        "ID": "041",
        "Sample_Type": "Logic_Trap",
        "Image_Path": "data/images/real_noon.jpg",
        "Text_Content": "深夜的街道格外宁静，月光洒满大地。",
        # GT: Ch3=1 (Logic Conflict)
        "GT_Final_Label": 1,
        "GT_Ch1_Tamper": 0, 
        "GT_Ch2_Mismatch": 0,  # CLIP觉得"街道"和"街道"很配
        "GT_Ch3_Logic": 1,     # 重点检测对象
        "Tamper_Method": "None",
        # Meta 必须填写真实情况(Day)，才能检测出文本(Night)是撒谎
        "Meta_Time": "Day", "Meta_Weather": "Sunny", "Meta_Location": "Street",
        "Meta_Fact": "Noon", "Meta_Object": "Street",
        "Source_Note": "图是白天，文说深夜"
    },
    # Case B: 天气冲突 (Weather)
    {
        "ID": "042",
        "Sample_Type": "Logic_Trap",
        "Image_Path": "data/images/real_sunny_beach.jpg",
        "Text_Content": "狂风暴雨袭击了海岸线。",
        "GT_Final_Label": 1,
        "GT_Ch1_Tamper": 0, 
        "GT_Ch2_Mismatch": 0,
        "GT_Ch3_Logic": 1,
        "Tamper_Method": "None",
        "Meta_Time": "Day", "Meta_Weather": "Sunny", "Meta_Location": "Beach",
        "Meta_Fact": "Calm", "Meta_Object": "Sea",
        "Source_Note": "图是晴天，文说暴雨"
    }
]

# ============================================================================
# 3. 生成 Excel
# ============================================================================
df = pd.DataFrame(data)
df = df[columns_order]  # 强制列顺序

file_name = "Yuanjing_Data_Standard_Final.xlsx"

try:
    df.to_excel(file_name, index=False, engine='openpyxl')
    
    # 获取绝对路径
    file_path = os.path.abspath(file_name)
    
    print("=" * 60)
    print("[SUCCESS] 最终版数据表已生成!")
    print("=" * 60)
    print(f"文件路径: {file_path}")
    print("-" * 60)
    
    print("\n[INFO] 表头字段 (16列):")
    print("-" * 60)
    for i, col in enumerate(df.columns):
        print(f"  {chr(65+i) if i < 26 else 'A'+chr(65+i-26)}列: {col}")
    
    print("\n[INFO] 字段设计说明 (为概率统计优化):")
    print("-" * 60)
    print("   1. GT_Ch1_Tamper   -> 对应 P1 (篡改概率)")
    print("   2. GT_Ch2_Mismatch -> 对应 P2 (不符概率)")
    print("   3. GT_Ch3_Logic    -> 对应 P3 (冲突概率)")
    print("   4. Sample_Type     -> 用于生成分类型的准确率报告")
    
    print("\n[STATS] 数据统计:")
    print("-" * 60)
    print(f"  总样本数: {len(df)}")
    for sample_type in df['Sample_Type'].unique():
        count = len(df[df['Sample_Type'] == sample_type])
        print(f"  - {sample_type}: {count} 条")
    
    print("\n[PREVIEW] 数据预览:")
    print("-" * 60)
    preview_cols = ['ID', 'Sample_Type', 'GT_Ch1_Tamper', 'GT_Ch2_Mismatch', 'GT_Ch3_Logic']
    print(df[preview_cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("[TIP] 概率融合公式:")
    print("  P_final = 1 - (1-P1)*(1-P2)*(1-P3)")
    print("-" * 60)
    print("请团队成员严格按照 Sample_Type 分类填入后续数据。")
    print("=" * 60)
    
except ImportError:
    print("[ERROR] 'openpyxl' 模块未安装")
    print("   请运行: pip install openpyxl")
except Exception as e:
    print(f"[ERROR] 生成失败: {e}")
