#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
BASE_PATH = "/home/boat/proxyISP/DGC-Net/proxydgc_eval"  # Change this if your eval directories are somewhere else
DIR_PATTERN = re.compile(
    r"eval_sl_train_v16\.2-chroma-hpatchesv4_pooled480x640_allHomoRepeatedRaw_standardize_gradac8_(\d+)_HpatchesV4"
)
AEPE_FILENAME = "aepe.npy"
NUM_VIEWPOINTS = 5

def load_aepe(filepath):
    """Try to load AEPE values from .npy or .py array file."""
    try:
        return np.load(filepath, allow_pickle=True)
    except Exception:
        with open(filepath, "r") as f:
            content = f.read()
        arr_str = re.findall(r"[\d\.eE\+\-]+", content)
        return np.array([float(x) for x in arr_str])

def main():
    # === Step 1: Find all matching directories ===
    eval_dirs = []
    for name in os.listdir(BASE_PATH):
        full_path = os.path.join(BASE_PATH, name)
        if os.path.isdir(full_path):
            match = DIR_PATTERN.match(name)
            if match:
                iter_num = int(match.group(1))
                eval_dirs.append((iter_num, full_path))

    # Sort directories by iteration number
    eval_dirs.sort(key=lambda x: x[0])

    # === Step 2: Collect AEPE values ===
    iters = []
    viewpoint_scores = [[] for _ in range(NUM_VIEWPOINTS)]

    for iter_num, dir_path in eval_dirs:
        aepe_path = os.path.join(dir_path, AEPE_FILENAME)
        if os.path.exists(aepe_path):
            data = load_aepe(aepe_path)
            if len(data) == NUM_VIEWPOINTS:
                iters.append(iter_num)
                for i in range(NUM_VIEWPOINTS):
                    viewpoint_scores[i].append(data[i])
            else:
                print(f"[WARN] Skipping {dir_path}: expected {NUM_VIEWPOINTS} values, got {len(data)}")
        else:
            print(f"[WARN] {AEPE_FILENAME} not found in {dir_path}")

    if not iters:
        print("❌ No AEPE data found.")
        return

    # === Step 3: Plot AEPE curves ===
    plt.figure(figsize=(10, 6))
    for i in range(NUM_VIEWPOINTS):
        plt.plot(iters, viewpoint_scores[i], marker='o', label=f"Viewpoint {i+1}")

    plt.xlabel("Iteration")
    plt.ylabel("AEPE")
    plt.title("AEPE over Iterations for 5 Viewpoints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === Step 4: Save and show plot ===
    out_file = "aepe_plot.png"
    plt.savefig(out_file)
    print(f"✅ Plot saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()
