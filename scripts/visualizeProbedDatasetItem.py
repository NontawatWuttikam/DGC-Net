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

    # Convert source image to tensor and add batch dimension
    src_tensor = to_tensor(source_image).unsqueeze(0)  # [1, 3, 240, 240]

    # Convert correspondence map to grid for grid_sample
    grid = torch.tensor(correspondence_map, dtype=torch.float32).unsqueeze(0)  # [1, 240, 240, 2]
    print("grid max min", grid.max(), grid.min())

    # Apply grid_sample
    mapped = F.grid_sample(src_tensor, grid, mode='bilinear', align_corners=True)
    mapped_img = to_pil_image(mapped.squeeze(0).clamp(0, 1))
    print("mapped.shape", mapped.shape)

    # Convert target image and mapped image to tensors for blending
    target_tensor = to_tensor(target_image)
    mapped_tensor = to_tensor(mapped_img)

    # Resize if needed (in case dimensions don't match)
    if target_tensor.shape != mapped_tensor.shape:
        print("Warning: shape mismatch, resizing mapped image")
        mapped_tensor = F.interpolate(mapped_tensor.unsqueeze(0), size=target_tensor.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

    # Alpha blend the two images (50% each)
    overlap_tensor = (0.5 * target_tensor + 0.5 * mapped_tensor).clamp(0, 1)
    overlap_img = to_pil_image(overlap_tensor)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(source_image)
    axs[0].set_title("Source Image")
    axs[0].axis("off")

    axs[1].imshow(target_image)
    axs[1].set_title("Target Image")
    axs[1].axis("off")

    axs[2].imshow(mapped_img)
    axs[2].set_title("Mapped Source via Pyro")
    axs[2].axis("off")

    axs[3].imshow(overlap_img)
    axs[3].set_title("Overlap: Target + Mapped")
    axs[3].axis("off")

    # Save figure
    save_name = output_dir / f"{pkl_path.stem}_viz.png"
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
