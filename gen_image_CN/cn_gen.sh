# nohup bash cn_gen.sh > nohup_cn_gen.out &

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ymc2

python "./controlnet_gen.py"