# nohup bash 01_train_unet.sh > nohup_train_unet.out &
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ymc


python "../reconU.py" \
    --real_image_dir="./train/target/" \
    --composition_image_dir="./train/source/" \
    --decoder_weights_path="../__output__/02-1_controlnet_keypoint/condition_weight/20251025-0301_unetDecoder01_daytime/checkpoints/best_DREAMSIM_checkpoint/checkpoint-150_value-0.29048516/reconstruction_decoder.pt" \
    --output_dir="./output" \
    --epochs=5000 \
    --batch_size=8 \
    --lr=1e-5 \
    --unfreeze_decoder_from_epoch=1000 \
    # --use_all_region_from_epoch= \