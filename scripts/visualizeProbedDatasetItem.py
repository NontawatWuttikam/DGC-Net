"""
This script visualizes the probed dataset items by loading the .pkl files generated during probing.
To ensure the dataset is processed correctly from proxyopt refactoring.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
import shutil
# Directories
input_dir = Path("probe_output")
output_dir = Path("probed_data_viz")
if input_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)


# Load all .pkl files
for pkl_path in input_dir.glob("*.pkl"):
    data = pickle.load(open(pkl_path, "rb"))

    source_image = data["source_image"]  # (240, 240, 3), uint8
    target_image = data["target_image"]  # (H, W, 3), uint8
    correspondence_map = data["correspondence_map_pyro"][-1]  # (240, 240, 2), normalized
    print(source_image.shape, target_image.shape, correspondence_map.shape)
    print("correspondence_map min max", correspondence_map.min(), correspondence_map.max())

    # Denormalize correspondence map from [-1, 1] to pixel coordinates of target image
    H_tgt, W_tgt = target_image.shape[:2]
    correspondence_map_px = np.empty_like(correspondence_map)
    correspondence_map_px[..., 0] = ((correspondence_map[..., 0] + 1) * 0.5 * (W_tgt - 1))
    correspondence_map_px[..., 1] = ((correspondence_map[..., 1] + 1) * 0.5 * (H_tgt - 1))

    # Convert source image to tensor and add batch dimension
    src_tensor = to_tensor(source_image).unsqueeze(0)  # [1, 3, 240, 240]

    # Convert correspondence map to grid for grid_sample
    grid = torch.tensor(correspondence_map, dtype=torch.float32).unsqueeze(0)  # [1, 240, 240, 2]

    # grid_sample expects grid in [-1, 1], and source image as CHW
    # Since we're mapping *source image to target coords*, we use grid_sample
    print("grid max min", grid.max(), grid.min())
    mapped = F.grid_sample(src_tensor, grid, mode='bilinear')
    mapped_img = to_pil_image(mapped.squeeze(0).clamp(0, 1))
    print("mapped.shape", mapped.shape)


    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(source_image)
    axs[0].set_title("Source Image")
    axs[0].axis("off")

    axs[1].imshow(target_image)
    axs[1].set_title("Target Image")
    axs[1].axis("off")

    axs[2].imshow(mapped_img)
    axs[2].set_title("Mapped Source via Pyro")
    axs[2].axis("off")

    # Save figure
    save_name = output_dir / f"{pkl_path.stem}_viz.png"
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
