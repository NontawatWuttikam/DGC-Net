source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgcnet
CUDA_VISIBLE_DEVICES=0 python train.py --image-data-path data \
                --csv-path-train data/csv/homo_aff_tps_train_tinyTimeMachine.csv \
                --csv-path-test data/csv/homo_aff_tps_train_tinyTimeMachine.csv