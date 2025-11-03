#!/bin/bash

# proxyopt config path
proxyoptConfig="/home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml"
pretrained="model/pretrained_models/dgc/checkpoint.pth"
# optimized hype path, specify "original" if wanted original hype rather than optimized hype according to proxyopt config file.
# stage2Checkpoint="/home/boat/proxyISP/DGC-Net/proxydgc_logs/train_v16.2-chroma-HumanTunedInitialHype_hpatchesv4_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac102/checkpoints/checkpoint_525900.pkl"
stage2Checkpoint="original"
extraSuffix="_HpatchesV4"
gpu_devices="0"
hpatchesSeqPrefix="sl" # ll, wl, sl
metrics=("aepe") # "aepe" "pck"

PreHPatchesPath="/mnt/ssd2tb/boat/thesis/s21fe_hpatches_v4"
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
python make_hpatches.py "$proxyoptConfig" "$stage2Checkpoint" "$hpatchesSeqPrefix" "$PreHPatchesPath"
conda deactivate

conda activate dgcnet
cd "$currentDir"
echo "$(pwd)"
python make_hpatches_csv.py "$HPatchesBasePath/HPatches" "$csvDir/proxyopt_hpatches"


# for each metric in ["aepe", "pck"]
# for metric in "aepe" "pck"; do
for metric in "${metrics[@]}"; do
    echo "Evaluating with metric: $metric"
    CUDA_VISIBLE_DEVICES="$gpu_devices" python eval.py \
        --image-data-path "$HPatchesBasePath/HPatches" \
        --csv-path "$csvDir/proxyopt_hpatches" \
        --pretrained "$pretrained" \
        --output-dir "proxydgc_eval/$dataName" \
        --metric "$metric"
done

# pytorch-superpoint/datasets/HPatches