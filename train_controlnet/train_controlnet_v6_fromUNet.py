#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Base Code
# https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py

import cv2
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
import random

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from datasets import Image as ds_Image
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

# OSError: broken data stream when reading image fileの回避
ImageFile.LOAD_TRUNCATED_IMAGES = True

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import util
import kwcoco_v2
from kwcoco_v2 import COCO_dataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__)


class ReconstructionDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Input is (N, in_channels, H/8, W/8) for resolution 512
        # Output should be (N, out_channels, H, W)
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'), # H/4
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'), # H/2
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'), # H
            torch.nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    val_target_coco = COCO_dataset(args.validation_target_coco)
    val_source_coco = COCO_dataset(args.validation_source_coco)
    
    validation_image_gids = random.sample(val_target_coco.get_imgId_list(), args.num_validation_images)
    
    validation_target_img_paths = [val_target_coco.image(gid).get_filePath() for gid in validation_image_gids]
    validation_source_img_paths = [val_source_coco.image(gid).get_filePath() for gid in validation_image_gids]
    validation_generate_imgs    = []
    validation_prompts = [val_target_coco.kw_dataset.imgs[gid]["prompt"] for gid in validation_image_gids]

    image_logs = []
    
    for gid, target_img_path, source_img_path, prompt in tqdm(zip(validation_image_gids, validation_target_img_paths, validation_source_img_paths, validation_prompts), desc="validation-gen", total=len(validation_prompts)):
        
        souce_img = Image.open(source_img_path).convert("RGB")
        target_img = Image.open(target_img_path).convert("RGB")
        
        gen_imgs = []
        
        for _ in range(args.num_validation_gen_images):
            with torch.autocast("cuda"):
                gen_image = pipeline(
                    prompt, souce_img, num_inference_steps=20, generator=generator
                ).images[0]
            
            gen_imgs.append(gen_image)
            validation_generate_imgs.append(gen_image)
            
        
        # Full-Reconstruction Metrics
        fr_iqa_metrics = {}
        if len(args.fr_metrics) > 0:
            fr_iqa_metrics = {key: None for key in args.fr_metrics}
            temp_similarities = {key: None for key in args.fr_metrics_calc_types}
            for gen_img in gen_imgs:
                # normal similarity
                if "normal" in args.fr_metrics_calc_types:
                    similary_calculator = util.ImageSimilarityCalculator(target_img, gen_img)
                    temp_similarities["normal"] = similary_calculator(types=args.fr_metrics)
                
                # object only similarity
                # COCOアノテーションから検出対象のみの画像を切り出す
                if "object" in args.fr_metrics_calc_types:
                    target_object_imgs, _   = val_target_coco.get_object_imgs(gid)
                    generate_object_imgs, _ = val_source_coco.get_object_imgs(gid, preprocess_func=lambda _: np.array(gen_img))
                    object_mean_similarities = []
                    for i, (t_img, g_img) in enumerate(zip(target_object_imgs, generate_object_imgs)):

                        # 画像のサイズが異なる場合は除外
                        if t_img.shape != g_img.shape:
                            continue
                        
                        # 画像の縦または横の長さがlimit_px以下の場合は除外
                        limit_px = 25
                        if t_img.shape[0]<limit_px or t_img.shape[1]<limit_px or g_img.shape[0]<limit_px or g_img.shape[1]<limit_px:
                            continue
                        
                        # その他エラーがあった場合は除外
                        try:
                            similary_calculator = util.ImageSimilarityCalculator(t_img, g_img)
                            object_mean_similarities += [similary_calculator(types=args.fr_metrics)]
                        except:
                            continue
                        
                        similary_calculator = util.ImageSimilarityCalculator(t_img, g_img)
                        object_mean_similarities += [similary_calculator(types=args.fr_metrics)]
                    
                    if len(object_mean_similarities) == 0:
                        temp_similarities["object"] = {metric:np.nan for metric in args.fr_metrics}
                    else:
                        temp_similarities["object"] = {metric:np.mean([value[metric] for value in object_mean_similarities]) for metric in object_mean_similarities[0].keys()}
                    
                
                # background noise similarity
                if "noise" in args.fr_metrics_calc_types:
                    target_object_img, _, _   = val_target_coco.get_object_and_background_imgs(gid, fill_noise=True)
                    generate_object_img, _, _ = val_source_coco.get_object_and_background_imgs(gid, fill_noise=True, preprocess_func=lambda _: np.array(gen_img))
                    
                    similary_calculator = util.ImageSimilarityCalculator(target_object_img, generate_object_img)
                    temp_similarities["noise"] = similary_calculator(types=args.fr_metrics)

                for sim_key in args.fr_metrics:
                    multi_metric = {}
                    for calc_type in args.fr_metrics_calc_types:
                        multi_metric[calc_type] = temp_similarities[calc_type][sim_key]
                    fr_iqa_metrics[sim_key] = multi_metric
                    
        image_logs.append(
            {
                "validation_target_image": target_img, 
                "validation_source_image": souce_img, 
                "validation_generate_images": gen_imgs, 
                "validation_prompt": prompt,
                "metrics": fr_iqa_metrics,
            }
        )
        
        
    # Distoribution-Based Metrics
    
    gen_dset_log = {}
    if len(args.db_metrics) > 0:
        target_image_dataset = util.ImageDataset(validation_target_img_paths, img_size=(args.resolution, args.resolution), data_size=args.num_validation_images)
        generate_image_dataset = util.ImageDataset([np.array(img) for img in validation_generate_imgs], img_size=(args.resolution, args.resolution), data_size=args.num_validation_images)
        
        db_iqa_metrics = {}
        db_similary_calculator = util.DatasetSimilarityCalculator(target_image_dataset, generate_image_dataset, dataset_batch_size=10)
        db_similarities = db_similary_calculator(types=args.db_metrics)
        for sim_key in args.db_metrics:
            db_iqa_metrics[sim_key] = db_similarities[sim_key]
        
        features_list = []
        features_list.append(util.get_Inceptionv3_features(target_image_dataset))
        features_list.append(util.get_Inceptionv3_features(generate_image_dataset))
        
        plotter = util.TensorPlotter(features_list, labels=["original", "generate"])
        plot_title_append_text = "\n"+f"[step-{step}]"+"\n"
        plot_title_append_text += "\n".join([f"{sim_key} : {str(db_iqa_metrics[sim_key])[:10]}" for sim_key in args.db_metrics])

        gen_dset_log["metrics"] = db_iqa_metrics
        for graph_type in args.plot_graph_types:
            gen_dset_log[graph_type] = plotter.get_plot_2d_3d(graph_type, plot_title_append=plot_title_append_text)
    
    
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":  #TODO: 必要ないかも？（見ないため、重いため）
            for log in image_logs:
                # validation_target_image = log["validation_target_image"]
                # validation_source_image = log["validation_source_image"]
                # validation_prompt = log["validation_prompt"]
                # generate_images = log["validation_generate_images"]

                # formatted_images = []

                # formatted_images.append(np.asarray(validation_target_image))
                # formatted_images.append(np.asarray(validation_source_image))

                # for gen_image in generate_images:
                #     formatted_images.append(np.asarray(gen_image))

                # formatted_images = np.stack(formatted_images)

                # tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
                pass
                
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs, gen_dset_log


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below."
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_target_coco",
        type=str,
        default=None,
        help=(
            "Path to a COCO json file containing validation target images."
            "(20231113-added)"
        ),
    )
    parser.add_argument(
        "--validation_source_coco",
        type=str,
        default=None,
        help=(
            "Path to a COCO json file containing validation source images."
            "(20231113-added)"
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--num_validation_gen_images",
        type=int,
        default=4,
        help=(
            "1枚の検証用画像に対して生成される生成画像の枚数"
            "(20231113-added)"
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--plot_graph_dir_name",
        type=str,
        default="plot_logs",
        help=(
            "added"
        ),
    )
    parser.add_argument(
        "--plot_graph_types",
        type=str,
        default=["pca", "tsne", "umap"],
        nargs="+",
        help=(
            f"`pca` or `tsne` or `umap`"
            "(20231111-added)"
        ),
    )
    parser.add_argument(
        "--image_log_dir_name",
        type=str,
        default="image_logs",
        help=(
            "added"
        ),
    )
    parser.add_argument(
        "--recon_decoder_log_dir_name",
        type=str,
        default="recon_decoder_logs",
        help=(
            "The directory where the reconstructed conditioning images will be saved. (added)"
        ),
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        default=None,
        nargs="+",
        help=(
            "added"
        ),
    )
    parser.add_argument(
        "--global_pool_conditions",
        action="store_true",
        help=(
            "added"
        ),
    )
    parser.add_argument(
        "--fr_metrics",
        type=str,
        default=[],
        nargs="+",
        help=(
            f"{util.evaluate_ImgImg_utils.ImageSimilarityCalculator.SUPPORTED_METRICS}"
            "(20231111-added)"
        ),
    )
    parser.add_argument(
        "--fr_metrics_calc_types",
        type=str,
        default=None,
        nargs="+",
        help=(
            "`normal` or `object` or `noise`"
            "(20231111-added)"
        ),
    )
    parser.add_argument(
        "--fr_metrics_save_model",
        type=str,
        default=None,
        help=(
            "`normal` or `object` or `noise`"
            "(20231111-added)"
        ),
    )
    parser.add_argument(
        "--db_metrics",
        type=str,
        default=[],
        nargs="+",
        help=(
            f"{util.evaluate_DatasetDataset_utils.DatasetSimilarityCalculator.SUPPORTED_METRICS}"
            "(20231113-added)"
        ),
    )
    parser.add_argument(
        "--reconstruction_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the reconstruction loss.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")
    
    if int(args.num_validation_images) * int(args.num_validation_gen_images) < 10:
        if len(args.db_metrics) > 0:
            raise ValueError("`--num_validation_images` * `--num_validation_gen_images` must be less than 10.")
        else:
            pass
            
    if args.validation_target_coco is None or args.validation_source_coco is None:
        raise ValueError("`--validation_target_coco` and `--validation_source_coco` must be specified.")
    
    if not Path(args.validation_target_coco).exists():
        raise ValueError("`--validation_target_coco` must be a valid path.")
    if not Path(args.validation_source_coco).exists():
        raise ValueError("`--validation_source_coco` must be a valid path.")
    
    if len(COCO_dataset(args.validation_target_coco).get_imgId_list()) < 10:
        raise ValueError("`--validation_target_coco` must contain less than 10 images.")
    if len(COCO_dataset(args.validation_source_coco).get_imgId_list()) < 10:
        raise ValueError("`--validation_source_coco` must contain less than 10 images.")
    
    if args.fr_metrics_calc_types is not None:
        if not args.fr_metrics_save_model in args.fr_metrics_calc_types:
            raise ValueError("`--fr_metrics_save_model` must be in `--fr_metrics_calc_types`.")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_train_dataset(args, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                data_files=args.train_data_files, 
                cache_dir=args.cache_dir,
            )

            def add_train_data_dir(example):
                example[args.image_column] = str(Path(args.train_data_dir)/str(example[args.image_column]))
                example[args.conditioning_image_column] = str(Path(args.train_data_dir)/str(example[args.conditioning_image_column]))
                return example
            
            
            dataset["train"] = dataset["train"].map(add_train_data_dir)
            dataset = dataset.cast_column( args.image_column , ds_Image(decode=True))
            dataset = dataset.cast_column( args.conditioning_image_column , ds_Image(decode=True))

        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if not (Path(args.output_dir)/Path(args.image_log_dir_name)).exists():
            (Path(args.output_dir)/Path(args.image_log_dir_name)).mkdir(exist_ok=True)
        if not (Path(args.output_dir)/Path(args.recon_decoder_log_dir_name)).exists():
            (Path(args.output_dir)/Path(args.recon_decoder_log_dir_name)).mkdir(exist_ok=True)
        if not (Path(args.output_dir)/Path(args.plot_graph_dir_name)).exists():
            (Path(args.output_dir)/Path(args.plot_graph_dir_name)).mkdir(exist_ok=True)
            for graph_type in args.plot_graph_types:
                (Path(args.output_dir)/Path(args.plot_graph_dir_name)/Path(graph_type)).mkdir(exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # 2023/09/14 add hirahara
    if args.global_pool_conditions:
        controlnet.config.global_pool_conditions = True

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            # Save controlnet
            for model in models:
                if isinstance(model, ControlNetModel):
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))
                elif isinstance(model, ReconstructionDecoder):
                    # This is a simple nn.Module, not a diffusers model.
                    torch.save(model.state_dict(), os.path.join(output_dir, "reconstruction_decoder.pt"))

            # Pop weights to signal to accelerator that we've handled saving
            while len(weights) > 0:
                weights.pop()

        def load_model_hook(models, input_dir):
            controlnet_model = None
            reconstruction_decoder_model = None
            
            for model in models:
                if isinstance(model, ControlNetModel):
                    controlnet_model = model
                elif isinstance(model, ReconstructionDecoder):
                    reconstruction_decoder_model = model

            if controlnet_model is not None:
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                controlnet_model.register_to_config(**load_model.config)
                controlnet_model.load_state_dict(load_model.state_dict())
                del load_model
            
            if reconstruction_decoder_model is not None and os.path.exists(os.path.join(input_dir, "reconstruction_decoder.pt")):
                reconstruction_decoder_model.load_state_dict(torch.load(os.path.join(input_dir, "reconstruction_decoder.pt")))

            # Pop models so that they are not loaded again
            while len(models) > 0:
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 20240101 added
    # The decoder of UNet2DConditionModel has out_channels=block_out_channels[-1] for SD v1/v2.
    reconstruction_decoder = ReconstructionDecoder(in_channels=unet.config.block_out_channels[-1])

    # Optimizer creation
    params_to_optimize = list(controlnet.parameters()) + list(reconstruction_decoder.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, tokenizer, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, reconstruction_decoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, reconstruction_decoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Hook to capture the output of the UNet's decoder. This will be used as input to the reconstruction decoder.
    captured_features = {}

    def get_unet_decoder_output_hook(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                captured_features[name] = output[0]
            else:
                captured_features[name] = output
        return hook

    hook_handle = unet.up_blocks[-1].register_forward_hook(get_unet_decoder_output_hook("decoder_output"))

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("train_data_files")
        tracker_config.pop("validation_target_coco")
        tracker_config.pop("validation_source_coco")
        tracker_config.pop("num_validation_gen_images")
        tracker_config.pop("fr_metrics")
        tracker_config.pop("fr_metrics_calc_types")
        tracker_config.pop("fr_metrics_save_model")
        tracker_config.pop("db_metrics")
        tracker_config.pop("plot_graph_dir_name")
        tracker_config.pop("plot_graph_types")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    checkpoint_save_list = {key : {
                                        "checkpoints_q":util.Custom_Q(args.checkpoints_total_limit), 
                                        "values_q":util.Custom_Q(args.checkpoints_total_limit)
                                    } 
                                for key in ["last"]+args.fr_metrics+args.db_metrics
                            }
    for epoch in range(first_epoch, args.num_train_epochs):
        last_reconstructed_cond_image = None
        last_target_cond_image = None
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual.
                # This call will also trigger the forward hook on unet.mid_block to capture its output.
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Reconstruct conditioning image from the UNet's decoder output
                unet_decoder_output = captured_features["decoder_output"]
                reconstructed_cond_image = reconstruction_decoder(unet_decoder_output.to(dtype=weight_dtype))

                # The conditioning image is RGB, but should be grayscale.
                # Convert to grayscale properly, not just by taking the R channel.
                target_cond_image = transforms.functional.rgb_to_grayscale(batch["conditioning_pixel_values"]).to(
                    device=reconstructed_cond_image.device, dtype=reconstructed_cond_image.dtype
                )

                # Heuristically detect and fix inverted conditioning images (e.g., black-on-white).
                # If an image is mostly bright, it's likely inverted. Invert it to be white-on-black,
                # which the subsequent masking and loss calculation expects.
                # This is done per-image in the batch.
                with torch.no_grad():
                    # Calculate mean brightness for each image in the batch
                    img_mean = torch.mean(target_cond_image, dim=(-1, -2), keepdim=True)
                    # Create a mask for images that are likely inverted (mean > 0.5)
                    inversion_mask = (img_mean > 0.5).float()
                    # Invert only the images identified by the mask
                    target_cond_image = (target_cond_image * (1 - inversion_mask)) + ((1.0 - target_cond_image) * inversion_mask)

                last_reconstructed_cond_image = reconstructed_cond_image
                last_target_cond_image = target_cond_image

                
                # Ignore black regions in loss, focusing on white regions.
                mask = (target_cond_image > 0.5).float()
                
                # Reconstruction loss (MSE loss on white areas)
                recon_loss = F.mse_loss(reconstructed_cond_image, target_cond_image, reduction="none")
                recon_loss = (recon_loss * mask).sum() / (mask.sum() + 1e-8)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = noise_loss + args.reconstruction_loss_weight * recon_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        image_logs, gen_dset_log = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        
                    if global_step % args.checkpointing_steps == 0:
                        last_model_save_path = Path(args.output_dir) / "checkpoints" / "last_checkpoint" / f"checkpoint-{global_step}"

                        accelerator.save_state(str(last_model_save_path))
                        # save Pipeline
                        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                                        args.pretrained_model_name_or_path,
                                        vae=vae,
                                        text_encoder=text_encoder,
                                        tokenizer=tokenizer,
                                        unet=unet,
                                        controlnet=controlnet,
                                        safety_checker=None,
                                        revision=args.revision,
                                        torch_dtype=weight_dtype,
                                    )
                        pipeline.to_json_file(last_model_save_path/"model_index.json")
                        logger.info(f"Saved state to {str(last_model_save_path)}")

                        if last_reconstructed_cond_image is not None:
                            num_images_to_save = 4
                            num_images_to_save = min(num_images_to_save, last_target_cond_image.shape[0])

                            target_images = last_target_cond_image.detach().cpu().float()
                            recon_images = last_reconstructed_cond_image.detach().cpu().float()

                            pil_images = []
                            to_pil = transforms.ToPILImage()

                            for i in range(num_images_to_save):
                                target_pil = to_pil(target_images[i])
                                recon_pil = to_pil(recon_images[i])
                                pil_images.append(target_pil)
                                pil_images.append(recon_pil)

                            if len(pil_images) > 0:
                                grid = image_grid(pil_images, rows=num_images_to_save, cols=2)
                                save_path = Path(args.output_dir) / args.recon_decoder_log_dir_name / f"step_{global_step}.png"
                                grid.save(str(save_path))

                        for label_key in checkpoint_save_list.keys():
                            if label_key == "last":
                                cp_res = checkpoint_save_list[label_key]["checkpoints_q"].enqueue(last_model_save_path)
                                if cp_res is not None:
                                    shutil.rmtree(str(cp_res))
                                continue

                            if label_key in args.fr_metrics:
                                metric_mean = np.nanmean([log["metrics"][label_key][args.fr_metrics_save_model] for log in image_logs])
                                metric_good_symbol = util.ImageSimilarityCalculator.metrics_annotation(label_key)["symbol"]
                            elif label_key in args.db_metrics:
                                metric_mean = gen_dset_log["metrics"][label_key]
                                metric_good_symbol = util.DatasetSimilarityCalculator.metrics_annotation(label_key)["symbol"]
                            else:
                                raise ValueError(f"Unknown label_key {label_key}")
                            
                            model_save_path = Path(args.output_dir) / "checkpoints" / f"best_{label_key}_checkpoint" / f"checkpoint-{global_step}_value-{str(metric_mean)[:10]}"

                            if len(checkpoint_save_list[label_key]["values_q"]) > 0:
                                if (metric_good_symbol == "^") and (checkpoint_save_list[label_key]["values_q"].max < metric_mean):
                                    pass
                                elif (metric_good_symbol == "v") and (checkpoint_save_list[label_key]["values_q"].min > metric_mean):
                                    pass
                                else:
                                    continue
                            
                            shutil.copytree(str(last_model_save_path), str(model_save_path))

                            cp_res = checkpoint_save_list[label_key]["checkpoints_q"].enqueue(model_save_path)
                            val_res = checkpoint_save_list[label_key]["values_q"].enqueue(metric_mean)

                            if cp_res is not None:
                                shutil.rmtree(str(cp_res))
                                
            logs = {
                "global_step": global_step,
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                "noise_loss": noise_loss.detach().item(),
                "recon_loss": recon_loss.detach().item()
            }
            progress_bar.set_postfix(**logs)

            if (accelerator.sync_gradients) and (image_logs is not None) and (gen_dset_log is not None) and (global_step % args.validation_steps == 0):
                metrics_logs = {}
                
                for metric_key in args.fr_metrics:
                    temp = {f"{metric_key}/{k}_{calc_type}": {f'gen-{j}':v for j, v in enumerate([log["metrics"][metric_key][calc_type] for log in image_logs])} for k, calc_type in enumerate(args.fr_metrics_calc_types)}
                    metrics_logs = metrics_logs|temp
                    
                # Distoribution-Based Metrics
                if "metrics" in gen_dset_log.keys():
                    # Combine all logs
                    logs = logs | gen_dset_log["metrics"] | metrics_logs

            accelerator.log(logs, step=global_step)


            if (accelerator.sync_gradients) and (image_logs is not None) and (global_step % args.validation_steps == 0):
                
                # all-generate-images save
                save_dir = Path(args.output_dir)/"image_logs_all"/f"s-{global_step}"
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                for i, log in enumerate(image_logs):
                    target_img = log["validation_target_image"]
                    source_img = log["validation_source_image"]
                    generate_images = log["validation_generate_images"]
                    images = [target_img, source_img] + generate_images
                    img_save_path = save_dir/f"s-{global_step}_images-{i}.png"
                    image_grid(images, 1, len(images)).save(str(img_save_path))
                
                # Full-Reference Metrics
                validation_target_images = [log["validation_target_image"] for log in image_logs[:10]]
                validation_source_images = [log["validation_source_image"] for log in image_logs[:10]]
                generate_images = [g_img for log in image_logs[:10] for g_img in log["validation_generate_images"]]
                images = validation_target_images + validation_source_images + generate_images
                img_save_path = Path(args.output_dir)/Path(args.image_log_dir_name)/f"s-{global_step}_images_{0}.png"
                save_img = image_grid(images, (len(images)//len(validation_target_images)), len(validation_target_images))
                save_img.resize((round(save_img.width * 0.25), round(save_img.height * 0.25))).save(str(img_save_path))

                # Distribution-Based Metrics
                if "metrics" in gen_dset_log.keys():
                    for graph_type in args.plot_graph_types:
                        graph_output_path = Path(args.output_dir)/Path(args.plot_graph_dir_name)/graph_type/f"s-{global_step}_{graph_type}.png"
                        cv2.imwrite(str(graph_output_path), gen_dset_log[graph_type])

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    hook_handle.remove()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
