import torch
from PIL import Image
import json
import os
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel

# 设置使用的GPU
device_id = 1
torch.cuda.set_device(device_id)
device = torch.device(f"cuda:{device_id}")

# 创建结果文件夹
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)


# 加载模型函数
def load_models():
    # 加载微调后的 CLIP 模型
    clip_model = CLIPModel.from_pretrained("./fine_tuned_clip").to(device)

    # 加载微调后的 CLIP 处理器
    clip_processor = CLIPProcessor.from_pretrained("./fine_tuned_clip_processor")

    # VAE和UNet权重文件及配置
    with open("./stable-diffusion-v1-5/vae/vae_config.json", "r") as f:
        vae_config = json.load(f)
    vae_weights_path = "./stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin"
    vae = AutoencoderKL(**vae_config).to(device)
    vae.load_state_dict(torch.load(vae_weights_path, map_location=device))

    with open("./stable-diffusion-v1-5/unet/unet_config.json", "r") as f:
        unet_config = json.load(f)
    unet_weights_path = "./stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin"
    unet = UNet2DConditionModel(**unet_config).to(device)
    unet.load_state_dict(torch.load(unet_weights_path, map_location=device))

    print('UNet 和 VAE 加载完毕！开始生成图像！')
    return clip_model, clip_processor, vae, unet


def generate_pokemon_image(prompt, clip_model, clip_processor, vae, unet):
    # 使用 CLIP 处理器处理文本
    inputs = clip_processor(text=prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs).unsqueeze(1).to(device)

    # 初始化噪声
    latents = torch.randn((1, unet.in_channels, 16, 16)).to(device)

    # 使用DDIM调度器
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=10)  # 设置推理步骤数

    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(latents, t, encoder_hidden_states=text_features).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 使用VAE解码图像
    with torch.no_grad():
        image = vae.decode(latents / 0.18215).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in image]

    return pil_images[0]


if __name__ == "__main__":
    # 加载模型
    clip_model, clip_processor, vae, unet = load_models()
    # 测试生成图像
    with open("test.txt", "r", encoding="utf-8") as f:
        test_prompts = [line.strip() for line in f.readlines() if line.strip()]

    test_prompts = test_prompts[:10]

    # 生成并保存图像
    for i, prompt in enumerate(test_prompts):
        image = generate_pokemon_image(prompt, clip_model, clip_processor, vae, unet)
        image.save(os.path.join(result_dir, f"image_{i + 1}.png"))
        print(f"图像 {i + 1} 生成完成！")

    print("所有图像生成完成！")