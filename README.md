
# MultiChannel-Reasoning-System

## Project Goals and Introduction

This project aims to develop a multi-modal fake news detection system specifically designed for patent application scenarios. The system integrates three distinct reasoning channels—Physical, Semantic, and Logical—to analyze multimedia content comprehensively.

The core objective is to process input data (images and text) through these independent channels and aggregate the results to assess credibility and detect potential forgeries or inconsistencies.

## Directory Structure

```text
MultiChannel-Reasoning-System/
│
├── .gitignore                  # [配置] 忽略规则 (已配置好)
├── README.md                   # [文档] 项目说明书
├── requirements.txt            # [依赖] pip install -r requirement
│
├── main_demo.py                 # [入口] 总指挥脚本 (串联三个通道)
│
├── data/                       # [数据中心]
│   ├── images/                 # [图片仓] 所有的 .jpg / .png 都放这里
│   │   ├── ps_splicing_001.jpg
│   │   ├── aigc_eraser_001.jpg
│   │   └── ... (50-200张)
│   ├── Yuanjing_Data_Standard_v2.xlsx  # [核心] 最终版数据表 (由 create_excel_v2.py 生成)
│   └── create_excel_v2.py       # [工具] 生成Excel的脚本 (保留 v2，删掉 v1)
│
├── channel_1_forgery_detection/ # [通道一] 物理层
│   ├── detector.py              # [核心] 篡改检测代码
│   ├── weights/                 # [模型] 放 .pth 权重文件 (Git会忽略)
│   └── utils.py                 # (可选) 图像预处理工具函数
│
├── channel_2_consistency_clip/  # [通道二] 语义层
│   ├── matcher.py               # [核心] CLIP检测代码
│   └── clip_utils.py            # (可选) CLIP专用辅助函数
│
└── channel_3_logic_rules/       # [通道三] 逻辑层
    ├── reasoner.py              # [核心] 逻辑推理代码
    └── knowledge_base.json      # (可选) 如果关键词库太大，可以独立存JSON

```

---

## 各模块开发任务指南

以下是各模块的简要代码分析与任务说明。关于具体的算法实现细节、参数设置及理论依据，请务必查阅各模块目录下的 **PDF 指南手册 (Guide Book)**。

### 1. 总控模块 (Root)

* **脚本**: `main_demo.py`
* **任务**: 负责系统的整体调度。需要读取 `data/` 下的 Excel 数据表，循环处理每一个样本，依次调用三个通道的检测接口，最后加权汇总得出最终判定结果。

### 2. 数据中心 (Data Center)

* **目录**: `data/`
* **任务**:
* **图像存储**: 将所有测试图片集中存放于 `images/` 目录。
* **数据标准化**: 运行 `create_excel_v2.py` 生成标准化的 Excel 文件 (`Yuanjing_Data_Standard_v2.xlsx`)，确保每一行数据准确对应图片路径和文本描述。



### 3. 通道一：物理层伪造检测 (Channel 1)

* **目录**: `channel_1_forgery_detection/`
* **核心代码**: `detector.py`
* **任务简述**: 负责图像层面的硬核防伪。需要编写代码加载 `weights/` 下的预训练模型，对输入图像进行信号分析（如噪声指纹、频谱异常）。
* **输入输出**: 输入图片路径，输出伪造置信度。
* **注意**: 详细的模型架构与训练细节请参阅该目录下的 **PDF 文档**。

### 4. 通道二：语义一致性检测 (Channel 2)

* **目录**: `channel_2_consistency_clip/`
* **核心代码**: `matcher.py`
* **任务简述**: 负责检测“图文不符”。利用 CLIP 模型提取图像和文本的特征向量，计算余弦相似度。
* **输入输出**: 输入图片与对应文本，输出一致性得分。
* **注意**: CLIP 模型的具体调用方式与阈值设定请参阅该目录下的 **PDF 文档**。

### 5. 通道三：逻辑规则推理 (Channel 3)

* **目录**: `channel_3_logic_rules/`
* **核心代码**: `reasoner.py`
* **任务简述**: 负责文本层面的逻辑自洽性检查。基于 `knowledge_base.json` 中的知识或预设逻辑规则，检测新闻要素是否存在冲突。
* **输入输出**: 输入文本信息，输出逻辑可信度。
* **注意**: 具体的逻辑规则定义与推理树结构请参阅该目录下的 **PDF 文档**。

---
