#!/bin/bash

PROXY_DGC_CONFIG="proxydgc_config/train_config1.yaml"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgcnet

# Run CSV generation and capture output path
CSV_PATH=$(python processCSV.py \
    --csv-path data/csv/homo_aff_tps_train.csv \
    --proxydgc-config "$PROXY_DGC_CONFIG" \
    --seed 999)

BATCH_SIZE=$(yq ".batch_size" $PROXY_DGC_CONFIG)
# Now run training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --proxydgc-config "$PROXY_DGC_CONFIG" \
    --image-data-path data \
    --csv-path-train "$CSV_PATH" \
    --csv-path-test "$CSV_PATH" \
    --batch-size "$BATCH_SIZE"
