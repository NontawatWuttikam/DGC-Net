#!/bin/bash
start_number=0
end_number=7500
step_size=100

echo "Running evaluation sweep from checkpoint $start_number to $end_number with step size $step_size"

# proxyopt config path
# proxydgc_eval_dir="proxydgc_eval_CMAES_train_lowlight_maxstd0.1_csadampfac1.5"
proxydgc_eval_dir="proxydgc_eval_FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_lowlight_pooled480x640_allHomoRepeatedRaw_standardize_lr0.005_gradac4"
proxyoptConfig="/home/boat/proxyISP/ProxyOpt/train_configs/v16.2-chroma-HumanTunedInitialHype.yaml"
pretrained="model/pretrained_models/dgc/checkpoint.pth"
# Base path for checkpoints
# stage2CheckpointBase="/home/boat/proxyISP/DGC-Net/proxydgc_logs/CMAES_lowlight_maxstd0.1_CSA1.5_noiseaug0.6/cma_checkpoints"
stage2CheckpointBase="/home/boat/proxyISP/DGC-Net/proxydgc_logs/FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_lowlight_pooled480x640_allHomoRepeatedRaw_standardize_lr0.005_gradac4/checkpoints"
# stage2Checkpoint="original"
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

mkdir -p "$proxydgc_eval_dir"

# Loop through checkpoint numbers
for checkpoint_num in $(seq $start_number $step_size $end_number); do
    echo "=================================================="
    echo "Processing checkpoint: $checkpoint_num"
    echo "=================================================="

    # Set the current checkpoint path
    stage2Checkpoint="${stage2CheckpointBase}/checkpoint_${checkpoint_num}.pkl"

    # Check if checkpoint file exists
    if [ ! -f "$stage2Checkpoint" ]; then
        echo "Warning: Checkpoint file not found: $stage2Checkpoint"
        echo "Skipping checkpoint $checkpoint_num"
        continue
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
            --output-dir "$proxydgc_eval_dir/$dataName" \
            --metric "$metric"
    done

    echo "Completed evaluation for checkpoint $checkpoint_num"
    echo ""
done

echo "All checkpoint evaluations completed!"
# pytorch-superpoint/datasets/HPatches