import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AdamW
from torch.utils.data import DataLoader
from dataloader import get_dataset, get_clip_processor
import os


# 数据加载与处理函数
def create_collate_fn(clip_processor):
    def collate_fn(batch):
        # 从批次中提取文本和图像
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]
        # 使用 CLIP 处理器进行预处理
        inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        # 将预处理后的数据移到 GPU 上
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        return inputs

    return collate_fn


# 对比损失函数
def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2.0


# 微调 CLIP 模型函数
def train_clip_model(model_name="clip_model", clip_processor=None, epochs=10, batch_size=8):
    # 加载数据集
    dataset = get_dataset()
    # 如果 clip_processor 未提供，则从模型名加载
    if clip_processor is None:
        clip_processor = get_clip_processor(model_name)

    # 加载本地预训练的 CLIP 模型
    clip_model = CLIPModel.from_pretrained(model_name).to("cuda")

    # 创建数据加载器
    collate_fn = create_collate_fn(clip_processor)
    dataloader = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # 使用 AdamW 优化器
    optimizer = AdamW(clip_model.parameters(), lr=5e-5)

    # 训练模型
    clip_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # 前向传播
            inputs = batch
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            # 计算对比损失
            loss = contrastive_loss(logits_per_image, logits_per_text)
            epoch_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")

    # 保存微调后的模型和处理器
    save_dir = "fine_tuned_clip"
    os.makedirs(save_dir, exist_ok=True)

    try:
        clip_model.save_pretrained(save_dir)
        clip_processor.save_pretrained(f"{save_dir}_processor")
        print(f"Model and processor saved successfully in {save_dir}")
    except Exception as e:
        print(f"Failed to save the model and processor: {e}")


if __name__ == "__main__":
    model_name = "clip_model"
    clip_processor = get_clip_processor(model_name)
    train_clip_model(model_name=model_name, clip_processor=clip_processor)
