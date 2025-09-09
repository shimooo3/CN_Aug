from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
import torch
from diffusers.utils import load_image
import os
from tqdm import tqdm

import sys
# sys.path.append('/home/shimoguchi/spring/tools/gen_image_controlnet/green_ratio')

# Define base directories
base_input_dir = "../images/gen_data/"
base_output_dir = "../images/gen_data/output/0722/green_ratio/"
os.makedirs(base_output_dir, exist_ok=True)

controlnet = [
    ControlNetModel.from_pretrained("../../../AI_aug/ControlNet_Plus_Plus/train/output/checkpoint-1000/controlnet/", torch_dtype=torch.float16),
]
# unet = UNet2DConditionModel.from_pretrained("../../diffusers/examples/text_to_image/grape_SD/checkpoint-11000/", subfolder="unet", torch_dtype=torch.float16)

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompts = [
    "a beautiful prompt"
]

negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

image_count = 1
max_images = 500  # 最大画像生成数を50に設定

for i in range(1, 2):
    if image_count > max_images:
        break  # 50枚を超えたらディレクトリループを終了
        
    dir_suffix = f"0{i}"
    dir2 = os.path.join(base_input_dir, dir_suffix, "overall")
    output_dir = os.path.join(base_output_dir, dir_suffix)

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(dir2)):
        if image_count > max_images:
            break  # 50枚を超えたらファイルループを終了
            
        image2_path = os.path.join(dir2, filename)
        image2 = load_image(image2_path)
        images = [image2]

        seed = image_count
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Select prompt based on image_count
        prompt = prompts[(image_count - 1) % len(prompts)]

        output_image = pipe(
            prompt,
            images,
            num_inference_steps=20,
            generator=generator,
            negative_prompt=negative_prompt,
        ).images[0]

        output_image_path = os.path.join(output_dir, filename)
        output_image.save(output_image_path)
        image_count += 1
        
        # 処理した画像数を表示
        print(f"Processed {image_count-1} images out of {max_images}")