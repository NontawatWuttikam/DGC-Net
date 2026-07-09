#!/bin/bash

# proxyopt config path
proxyoptConfig="/home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml"
pretrained="model/pretrained_models/dgc/checkpoint.pth"
# optimized hype path, specify "original" if wanted original hype rather than optimized hype according to proxyopt config file.
# stage2Checkpoint="/home/boat/proxyISP/DGC-Net/proxydgc_logs/FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_sunlit_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32/checkpoints/checkpoint_105000.pkl"
# stage2Checkpoint="original"
# stage2Checkpoint="/home/boat/proxyISP/ProxyOpt/replication_adjusted/v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_lowlight_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8/original_color_hype/checkpoint_45000.pkl"
# stage2Checkpoint="/home/boat/proxyISP/ProxyOpt/replication_output/v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8/checkpoints/checkpoint_120000.pkl"
# stage2Checkpoint="/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_lowlight_maxstd0.01_CSA10.0_noiseaug0.6/cma_checkpoints/checkpoint_2050.pkl"
# stage2Checkpoint="/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_sunlit_maxstd0.01_CSA10.0_noiseaug0.6/cma_checkpoints/checkpoint_2040.pkl"
stage2Checkpoint="/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_lowlight_maxstd0.01_CSA10.0_noiseaug0.6/cma_checkpoints/checkpoint_1859.pkl"
# stage2Checkpoint="/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_lowlight_maxstd0.1_CSA1.5_noiseaug0.6/cma_checkpoints/checkpoint_1859.pkl"
extraSuffix="_HpatchesV4.1"
gpu_devices="0"
hpatchesSeqPrefix="ll" # ll, wl, sl
metrics=("aepe") # "aepe" "pck"

PreHPatchesPath="/mnt/ssd2tb/boat/thesis/s21fe_hpatches_v4.1"
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