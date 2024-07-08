from datasets import load_dataset
from transformers import CLIPProcessor


# 加载本地数据集函数
def get_dataset():
    # 从本地加载数据集
    dataset = load_dataset('parquet', data_files={
        'train': './data/train-00000-of-00001-a1df3b486d3a28b0.parquet'
    })
    return dataset


# 获取 CLIP 处理器函数
def get_clip_processor(model_name="clip_model"):
    # 加载本地 CLIP 处理器（用于文本和图像预处理）
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_processor
