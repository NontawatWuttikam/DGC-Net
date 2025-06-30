#!/bin/bash

# proxyopt config path
proxyoptConfig="../ProxyOpt/train_configs/v16.1.yaml"
pretrained="model/pretrained_models/dgc/checkpoint.pth"
# optimized hype path, specify "original" if wanted original hype rather than optimized hype according to proxyopt config file.
stage2Checkpoint="original"
# stage2Checkpoint="original"
extraSuffix=""
gpu_devices="0"
hpatchesSeqPrefix="sl" # ll, wl, sl
metrics=("aepe" "pck")

HPatchesBasePath="/home/boat/proxyISP/pytorch-superpoint/datasets"  # Adjust this path as needed
superpointBasePath="/home/boat/proxyISP/pytorch-superpoint"
csvDir="data/csv"

source ~/miniconda3/etc/profile.d/conda.sh
currentDir=$(pwd)
if [ "$hpatchesSeqPrefix" == "" ]; then
    hpatchesSeqPrefix="all"
fi

# Determine dataName based on stage2Checkpoint
if [ "$stage2Checkpoint" == "original" ]; then
    dataName="eval_${hpatchesSeqPrefix}_$(basename "$proxyoptConfig" .yaml)_original"
else
    parent_dir=$(basename "$(dirname "$(dirname "$stage2Checkpoint")")")
    checkpoint_name=$(basename "$stage2Checkpoint")
    step_number="${checkpoint_name%.pkl}"
    step_number="${step_number#checkpoint_}"
    dataName="eval_${hpatchesSeqPrefix}_${parent_dir}_${step_number}"
fi

dataName="${dataName}${extraSuffix}"
echo "Using dataName: $dataName"
# Clean up old dataset
rm -rf "$HPatchesBasePath/HPatches"

# Activate environment and run hpatches generation
conda activate proxyopt
cd "$superpointBasePath"
python make_hpatches.py "$proxyoptConfig" "$stage2Checkpoint" "$hpatchesSeqPrefix"
conda deactivate

conda activate dgcnet
cd "$currentDir"
echo "$(pwd)"
python make_hpatches_csv.py "$HPatchesBasePath/HPatches" "$csvDir/proxyopt_hpatches"


# for each metric in ["aepe", "pck"]
# for metric in "aepe" "pck"; do
for metric in "${metrics[@]}"; do
    echo "Evaluating with metric: $metric"
    python eval.py \
        --image-data-path "$HPatchesBasePath/HPatches" \
        --csv-path "$csvDir/proxyopt_hpatches" \
        --pretrained "$pretrained" \
        --output-dir "proxydgc_eval/$dataName" \
        --metric "$metric"
done

# pytorch-superpoint/datasets/HPatches