#!/bin/bash

# Default pretrained and GPU
pretrained="model/pretrained_models/dgc/checkpoint.pth"
gpu_devices="0"
metrics=("aepe" "pck") # "aepe" "pck"

# Paths
PreHPatchesPath="/mnt/ssd2tb/boat/thesis/s21fe_hpatches_v4.1"
HPatchesBasePath="/home/boat/proxyISP/pytorch-superpoint/datasets"  # Adjust this path as needed
superpointBasePath="/home/boat/proxyISP/pytorch-superpoint"
HPatchesCacheRoot="/home/boat/proxyISP/pytorch-superpoint/datasets/HPatches_caches_DGC"
csvDir="data/csv"
extraSuffix="_HpatchesV4.1"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
currentDir=$(pwd)

# List of checkpoints with individual config and prefix
# Format: "checkpoint_path proxyoptConfig hpatchesSeqPrefix"
checkpoints=(
    # "original /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml sl"
    # "original /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml ll"

    # "/home/boat/proxyISP/ProxyOpt/replication_output/v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8/checkpoints/checkpoint_120000.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml sl"
    # "/home/boat/proxyISP/ProxyOpt/v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_defaultcolorhuesat/checkpoints/checkpoint_120000.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml sl"
    # "/home/boat/proxyISP/ProxyOpt/replication_output/v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_denoise3/checkpoints/checkpoint_120000.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml ll"
    # "/home/boat/proxyISP/ProxyOpt/v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_lowlight_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_defaultcolorhuesat_denoise/checkpoints/checkpoint_45000.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-ISPDefaultInitialHype.yaml ll"

    # "/home/boat/proxyISP/DGC-Net/proxydgc_logs/FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_sunlit_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32/checkpoints/checkpoint_105000.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-HumanTunedInitialHype.yaml sl"
    # "/home/boat/proxyISP/DGC-Net/proxydgc_logs/FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_lowlight_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32/checkpoints/checkpoint_123000.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-HumanTunedInitialHype.yaml ll"

    # cma
    "/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_sunlit_maxstd0.01_CSA10.0_noiseaug0.6/cma_checkpoints/checkpoint_2040.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-HumanTunedInitialHype.yaml sl"
    "/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_lowlight_maxstd0.01_CSA10.0_noiseaug0.6/cma_checkpoints/checkpoint_2300.pkl /home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-HumanTunedInitialHype.yaml ll"
)

# Loop over each checkpoint
for entry in "${checkpoints[@]}"; do
    # Split entry into variables
    IFS=' ' read -r stage2Checkpoint proxyoptConfig hpatchesSeqPrefix <<< "$entry"

    echo "Processing checkpoint: $stage2Checkpoint"
    echo "Using config: $proxyoptConfig"
    echo "HPatches prefix: $hpatchesSeqPrefix"

    # Determine dataName
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

    # Clean old dataset
    rm -rf "$HPatchesBasePath/HPatches"

    # Make HPatches
    conda activate proxyopt
    cd "$superpointBasePath"
    python make_hpatches.py "$proxyoptConfig" "$stage2Checkpoint" "$hpatchesSeqPrefix" "$PreHPatchesPath"
    conda deactivate

    # Cache generated HPatches for DGC-Net
    mkdir -p "$HPatchesCacheRoot"

    cacheDest="$HPatchesCacheRoot/$dataName"
    echo "Caching HPatches to: $cacheDest"

    # Remove existing cache if present (optional but safer)
    rm -rf "$cacheDest"

    cp -a "$HPatchesBasePath/HPatches" "$cacheDest"

    # Make CSV
    conda activate dgcnet
    cd "$currentDir"
    python make_hpatches_csv.py "$HPatchesBasePath/HPatches" "$csvDir/proxyopt_hpatches"

    # Run evaluation for each metric
    for metric in "${metrics[@]}"; do
        echo "Evaluating with metric: $metric"
        CUDA_VISIBLE_DEVICES="$gpu_devices" python eval.py \
            --image-data-path "$HPatchesBasePath/HPatches" \
            --csv-path "$csvDir/proxyopt_hpatches" \
            --pretrained "$pretrained" \
            --output-dir "proxydgc_eval/$dataName" \
            --metric "$metric"
    done
done
