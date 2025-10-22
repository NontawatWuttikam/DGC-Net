import os
import numpy as np
import matplotlib.pyplot as plt

# Set the output filename
output_file = 'comparison_pck_plot.png'

# Define evaluation configurations
evals = [
    {
        "path": "proxydgc_eval/eval_sl_v16.1_original",
        "label": "Original Hype"
    },
    {
        "path": "proxydgc_eval/eval_sl_train_sunlit_pooled480x640_gradac1_allHomoRepeatedRaw_standardize_130000",
        "label": "Optimized Sunlit"
    }
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
fig, axes = plt.subplots(1, 5, figsize=(20,5), sharex=True)
x = np.arange(200)

for i in range(5):
    ax = axes[i]
    for pck, label in zip(pck_list, labels):
        ax.plot(x, pck[i], label=label)
    title = f"H-1-{i+2}"
    ax.set_title(title)
    ax.set_ylabel('PCK')
    ax.set_xlabel('threshold')
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')

axes[-1].set_xlabel('Frame Index')
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Figure saved to {output_file}")
