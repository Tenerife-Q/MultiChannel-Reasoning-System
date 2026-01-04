import pandas as pd
import os

# 1. 严格按照要求的 10 个字段定义列名顺序 (A列 -> J列)
columns_order = [
    "ID", 
    "Image_Path", 
    "Text_Content", 
    "GT_Final_Label", 
    "Ch1_Tamper_Label", 
    "Ch2_Consis_Label", 
    "Meta_Time", 
    "Meta_Scene", 
    "Meta_Object", 
    "Source_Note"
]

# 2. 准备数据：严格对应你提供的 3 条示例
# 注意：为了对应 Excel 格式，数字标签使用整数，文本保持原样
data = [
    # 第 1 行：图片篡改数据 (P图)
    {
        "ID": "001",
        "Image_Path": "./data/img1_p_graph.jpg",
        "Text_Content": "两国领导人在峰会上亲切握手。",
        "GT_Final_Label": "Fake",
        "Ch1_Tamper_Label": 1,  # 重点：图是假的
        "Ch2_Consis_Label": 1,  # 图文看起来匹配
        "Meta_Time": "Day",
        "Meta_Scene": "Meeting",
        "Meta_Object": "Person", # 根据场景逻辑补全
        "Source_Note": "使用CASIA数据集拼接篡改图"
    },
    # 第 2 行：图文不符数据 (移花接木)
    {
        "ID": "002",
        "Image_Path": "./data/img2_fire.jpg",
        "Text_Content": "突发：昨晚市中心商场发生特大火灾。",
        "GT_Final_Label": "Fake",
        "Ch1_Tamper_Label": 0,  # 图是真的
        "Ch2_Consis_Label": 0,  # 重点：图文不符
        "Meta_Time": "Day",
        "Meta_Scene": "Forest",
        "Meta_Object": "Tree/Fire", # 根据场景逻辑补全
        "Source_Note": "真实森林火灾图配谣言文字"
    },
    # 第 3 行：逻辑冲突数据 (违反常识)
    {
        "ID": "003",
        "Image_Path": "./data/img3_sunny.jpg",
        "Text_Content": "台风‘梅花’刚刚登陆，狂风暴雨。",
        "GT_Final_Label": "Fake",
        "Ch1_Tamper_Label": 0,
        "Ch2_Consis_Label": 0,
        "Meta_Time": "Day",
        "Meta_Scene": "Sunny Sky",
        "Meta_Object": "Sky/Tree", # 根据场景逻辑补全
        "Source_Note": "天气属性逻辑冲突"
    }
]

# 3. 创建 DataFrame
df = pd.DataFrame(data)

# 4. 强制调整列顺序以符合 Excel 表头要求
df = df[columns_order]

# 5. 导出为 Excel 文件
file_name = "Yuanjing_Data_Standard_v1.xlsx"

try:
    # 检查是否安装了 openpyxl，如果没有请运行 pip install openpyxl
    df.to_excel(file_name, index=False, engine='openpyxl')
    
    # 获取绝对路径，方便查找
    file_path = os.path.abspath(file_name)
    print(f"Success: Excel file created successfully.")
    print(f"Path: {file_path}")
    print("-" * 30)
    print("Verification of Columns:")
    print(df.columns.tolist())
    
except ImportError:
    print("Error: 'openpyxl' module is missing. Please run: pip install openpyxl")
except Exception as e:
    print(f"Error occurred: {e}")