# nohup bash 02_controlnet_train_keypoint_v5.sh > nohup_trak_controlnet.out &
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ymc

learning_rate="5e-5"
train_data_dir="../__dataset__/04-1_controlnet_keypoint/train/train/"
train_data_num=$(wc -l < "${train_data_dir}/prompt.json")

today=`date "+%Y%m%d-%H%M"`
output_dir_name="../__output__/02-1_controlnet_keypoint/condition_weight/_${today}"

validation_checkpoints_steps=50

accelerate launch "../train_controlnet_v5.py" \
    --controlnet_model_name_or_path="../__model__/01_controlnet_canny/20231229-0049_3e-4/checkpoints/best_LPIPS_SQ_checkpoint/checkpoint-14800_value-0.12218771/controlnet/" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --output_dir=$output_dir_name \
    \
    --train_data_dir=$train_data_dir \
    --train_data_files="prompt.json" \
    --image_column="target" \
    --conditioning_image_column="source" \
    --caption_column="prompt" \
    \
    --validation_steps=$validation_checkpoints_steps \
    --validation_target_coco "../__dataset__/04-1_controlnet_keypoint/valid_coco/target/valid.json" \
    --validation_source_coco "../__dataset__/04-1_controlnet_keypoint/valid_coco/source/valid.json" \
    --num_validation_images=50 \
    --num_validation_gen_images=1 \
    --fr_metrics "LPIPS_SQ" "SSIM" "DREAMSIM"\
    --fr_metrics_calc_types "normal" "object" "noise"\
    --fr_metrics_save_model "object"\
    --db_metrics "FID" "KID"\
    --plot_graph_types "pca" "tsne"\
    --max_train_steps=5000 \
    --train_batch_size=1 \
    --learning_rate=$learning_rate \
    \
    --checkpointing_steps=$validation_checkpoints_steps \
    --checkpoints_total_limit=3 \
    \
    --gradient_checkpointing \
    --gradient_accumulation_steps=4 \
    --resolution=512 \
    --use_8bit_adam \
    --seed=42 \


# cp "./nohup.out" "$output_dir_name/nohup.out"
# : > "./nohup.out"

# cp "$0" "$output_dir_name/$0"