# nohup bash mcn_gen.sh > nohup_mcn_gen.out &

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ymc2

python "./multi_controlnet_gen.py"