import os
import numpy as np
import matplotlib.pyplot as plt

# Set the output filename
output_file = 'comparison_pck_plot_wl_HPatchesV4.1.png'

# Define evaluation configurations
evals = [
    # {
    #     "path": "proxydgc_eval/eval_sl_v16.2-chroma-ISPDefaultInitialHype_original_HpatchesV4",
    #     "label": "Original Well-lit"
    # },
    # {
    #     "path": "proxydgc_eval/eval_sl_v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_120000_HpatchesV4",
    #     "label": "Visual-Optimized Well-lit"
    # },
    # {
    #     "path": "proxydgc_eval/eval_sl_FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_sunlit_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32_105000_HpatchesV4",
    #     "label": "Feature-Optimized Well-lit"
    # },
    # {
    #     "path": "proxydgc_eval/eval_ll_v16.2-chroma-ISPDefaultInitialHype_original_HpatchesV4",
    #     "label": "Original Low-light"
    # },
    # {
    #     "path": "proxydgc_eval/eval_ll_v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_lowlight_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_45000_HpatchesV4",
    #     "label": "Visual-Optimized Low-light"
    # },
    # {
    #     "path": "proxydgc_eval/eval_ll_FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_lowlight_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32_123000_HpatchesV4",
    #     "label": "Feature-Optimized Low-light"
    # },

    # HpatchesV4.1 ll
    # {
    #     "path": "proxydgc_eval/eval_ll_v16.2-chroma-ISPDefaultInitialHype_original_HpatchesV4.1",
    #     "label": "Orig-ISP Low-light"
    # },
    # {
    #     "path": "/home/boat/proxyISP/DGC-Net/proxydgc_eval/eval_ll_v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_lowlight_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_defaultcolorhuesat_denoise_45000_HpatchesV4.1",
    #     "label": "VO-ISP Low-light"
    # },
    # {
    #     "path": "proxydgc_eval/eval_ll_CMAES_lowlight_maxstd0.01_CSA10.0_noiseaug0.6_2300_HpatchesV4.1",
    #     "label": "FO-ISP-CMA Low-light"
    # },
    # {
    #     "path": "proxydgc_eval/eval_ll_FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_lowlight_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32_123000_HpatchesV4.1",
    #     "label": "FO-ISP-Proxy Low-light"
    # },

    {
        "path": "proxydgc_eval/eval_sl_v16.2-chroma-ISPDefaultInitialHype_original_HpatchesV4.1",
        "label": "Orig-ISP Well-lit"
    },
    {
        "path": "/home/boat/proxyISP/DGC-Net/proxydgc_eval/eval_sl_v16.2-chroma-HumanTunedInitialHype_replicate-s21fe_sunlit_lr0.0005_schedulerPlateauTo0.00001_bs1_ga8_adjust_defaultcolorhuesat_120000_HpatchesV4.1",
        "label": "VO-ISP Well-lit"
    },
        {
        "path": "proxydgc_eval/eval_sl_CMAES_sunlit_maxstd0.01_CSA10.0_noiseaug0.6_2040_HpatchesV4.1",
        "label": "FO-ISP-CMA Well-lit"
    },
    {
        "path": "proxydgc_eval/eval_sl_FIXZEROGRADBUG_CFANORMALIZE_train_v16.2-chroma-HumanTunedInitialHype_sunlit_pooled480x640_allHomoRepeatedRaw_standardize_lr0.0005_gradac32_105000_HpatchesV4.1",
        "label": "FO-ISP-Proxy Well-lit"
    },
    
    
]

# Initialize lists
pck_list = []
labels = []

# Load PCK data using unified structure
for eval_config in evals:
    eval_dir = eval_config["path"]
    custom_label = eval_config["label"]
    pck_path = os.path.join(eval_dir, 'pck.npy')

    if os.path.isfile(pck_path):
        try:
            pck = np.load(pck_path)
            if pck.shape == (5, 200):
                pck_list.append(pck)
                labels.append(custom_label)
            else:
                print(f"Skipping {eval_dir}: shape {pck.shape} is not (5, 200)")
        except Exception as e:
            print(f"Failed to read {pck_path}: {e}")
    else:
        print(f"{pck_path} not found")

if not pck_list:
    print("No valid pck.npy files found.")
    exit()

# Plotting
fig, axes = plt.subplots(1, 5, figsize=(20,3), sharex=True, sharey=True)
x = np.arange(200)
# set font to times new roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# change font of ticks to times new roman
for ax in axes:
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname("Times New Roman")

for i in range(5):
    ax = axes[i]
    for pck, label in zip(pck_list, labels):
        ax.plot(x, pck[i], label=label)
    title = f"Viewpoint {i+1}"
    ax.set_title(title, fontname="Times New Roman")
    ax.set_ylabel('PCK', fontname="Times New Roman")
    # ax.set_xlabel('threshold', fontname="Times New Roman")
    ax.grid(True)
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Figure saved to {output_file}")
