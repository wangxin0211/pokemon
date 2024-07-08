# Pokemon Text-to-Image Generation

## 实验简介
通过微调CLIP模型和使用Stable Diffusion模型实现根据文本描述生成宝可梦图像的功能。

## 文件结构
- `dataloader.py`: 数据加载与预处理
- `train_clip.py`: CLIP模型微调
- `generate_images.py`: 图像生成
- `requirements.txt`: 所需库
- `README.md`: 说明

## 依赖安装
```bash
pip install -r requirements.
```

## 实验步骤
1.使用Pokémon数据集对CLIP模型进行微调，使其更擅长理解宝可梦相关的文本和图像关系。

2.使用Stable Diffusion 1.5版本的预训练权重初始化VAE和UNet模型。

3.输入文本描述，利用微调后的CLIP模型将文本编码为嵌入向量。将这个嵌入向量输入到Stable Diffusion模型中，指导图像生成。

##实验结果
图片生成结果放在`./result/`

但是经过各种调参以及模型调整，都无法得到理想的实验结果。