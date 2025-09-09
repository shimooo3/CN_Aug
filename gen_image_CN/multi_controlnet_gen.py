from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
import os
import random
from tqdm import tqdm


# Define base directories
base_input_dir = "../images/gen_data/"
base_output_dir = "../images/gen_data/output/0722/green_ratio_mcn/"
os.makedirs(base_output_dir, exist_ok=True)

controlnet = [
    #line2line
    ControlNetModel.from_pretrained("../../AI_aug_gen/__output__/01_controlnet_canny/20240711-0318_3e-4/checkpoints/best_LPIPS_SQ_checkpoint/checkpoint-550_value-0.20505732/controlnet/", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("../../../AI_aug/ControlNet_Plus_Plus/train/output/checkpoint-1000/controlnet/", torch_dtype=torch.float16),
]

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)



pipe.enable_model_cpu_offload()

# prompt = "a photograph of grape cane in vine yard, ymc"
prompts = [
    "a beautiful prompt"
]
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

image_count = 1
max_images = 500

for i in range(1, 3):
    if image_count > max_images:
        break
    dir_suffix = f"0{i}"
    dir1 = os.path.join(base_input_dir, dir_suffix, "line2line")
    dir2 = os.path.join(base_input_dir, dir_suffix, "overall")
    output_dir = os.path.join(base_output_dir, dir_suffix)

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(dir1)):
        if filename in os.listdir(dir2):
            image1_path = os.path.join(dir1, filename)
            image2_path = os.path.join(dir2, filename)

            image1 = load_image(image1_path)
            image2 = load_image(image2_path)
            images = [image1, image2]

            #seed = random.randint(0, 1000000)
            seed = image_count
            
            generator = torch.Generator(device="cuda").manual_seed(seed)
            prompt = prompts[(image_count - 1) % len(prompts)]

            output_image = pipe(
                prompt,
                images,
                num_inference_steps=20,
                generator=generator,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=[0.2, 0.9],
            ).images[0]

            output_image_path = os.path.join(output_dir, filename)
            output_image.save(output_image_path)
            #print(f"Saved {output_image_path} with seed {seed}")
            image_count += 1
