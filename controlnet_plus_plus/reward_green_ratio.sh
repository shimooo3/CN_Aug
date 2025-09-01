# nohup bash reward_green_ratio.sh > green_ratio.out &

export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
export CONTROLNET_DIR="../raw_data/20231229-0049_3e-4/checkpoints/best_LPIPS_SQ_checkpoint/checkpoint-18500_value-0.12077894/controlnet/"
export REWARDMODEL_DIR="green_ratio"
export TRAIN_DATA_DIR="../train_data/"
export VALIDATION_DATA_DIR="../valid_data/"
export OUTPUT_DIR="./output"

accelerate launch --config_file "config.yml" \
 --main_process_port=23156 reward_control.py \
 --report_to="tensorboard" \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="green_ratio" \
 --seed=42 \
 --train_data_dir=$TRAIN_DATA_DIR \
 --validation_data_dir=$VALIDATION_DATA_DIR \
 --image_column="source" \
 --caption_column="prompt" \
 --conditioning_image_column="target" \
 --label_column="mask" \
 --resolution=512 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=8 \
 --max_train_steps=1500 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=0 \
 --checkpointing_steps=100 \
 --grad_scale=2.0 \
 --use_ema \
 --validation_steps=50 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200