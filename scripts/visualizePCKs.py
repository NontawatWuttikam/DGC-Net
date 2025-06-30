import os
import numpy as np
import matplotlib.pyplot as plt

# Set the output filename
output_file = 'comparison_pck_plot.png'  # Change this to whatever you want (e.g., .pdf, .svg)

# Set the root directory containing M subdirectories
eval_dirs = [
    # 'proxydgc_eval/eval_wl_train_welllit_pooled480x640_gradac1_allHomoRepeatedRaw_standardize_43300',
    # 'proxydgc_eval/eval_wl_train_welllit_pooled480x640_gradac1_allHomoRepeatedRaw_standardize_160000',
    # 'proxydgc_eval/eval_wl_train_welllit_pooled480x640_gradac1_allHomoRepeatedRaw_standardize_120000',
    # 'proxydgc_eval/eval_wl_v16.1_original'
    "proxydgc_eval/eval_sl_train_sunlit_pooled480x640_gradac1_allHomoRepeatedRaw_standardize_281500",
    "proxydgc_eval/eval_sl_train_sunlit_pooled480x640_gradac1_allHomoRepeatedRaw_standardize_120000",
    "proxydgc_eval/eval_sl_v16.1_original",
]

# Initialize a list to store all loaded PCK arrays
pck_list = []
labels = []

for eval_dir in eval_dirs:
    pck_path = os.path.join(eval_dir, 'pck.npy')
    if os.path.isfile(pck_path):
        try:
            pck = np.load(str(pck_path))
            if pck.shape == (5, 200):
                pck_list.append(pck)
                labels.append(os.path.basename(eval_dir))
            else:
                print(f"Skipping {eval_dir}: shape {pck.shape} is not (5, 200)")
        except Exception as e:
            print(f"Failed to read {pck_path}: {e}")
    else:
        print(f"{pck_path} not found")

if not pck_list:
    print("No valid pck.py files found.")
    exit()

# Plotting
fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
x = np.arange(200)

for i in range(5):
    ax = axes[i]
    for pck, label in zip(pck_list, labels):
        ax.plot(x, pck[i], label=label)
    ax.set_title(f'Keypoint {i+1}')
    ax.set_ylabel('PCK')
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')

axes[-1].set_xlabel('Frame Index')
plt.tight_layout()
plt.savefig(output_file, dpi=300)  # Save with high resolution
print(f"Figure saved to {output_file}")
