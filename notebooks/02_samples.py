# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: src
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sample Visualization
# Visual spot-check of images and masks from the val split.

# %% [markdown]
# ## Validation samples
# Uses `make_val_transforms` (squish resize to 512×512, ImageNet normalization).

# %%
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from canvit_probes.datasets.ade20k import (
    IGNORE_LABEL,
    ADE20kDataset,
    make_val_transforms,
)
from scipy.ndimage import label as cc_label

from src.io import ade20k_root

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

ResizeMode = Literal["center_crop", "squish"]


@dataclass
class Config:
    probe_repo: str
    ade20k_root: Path = field(default_factory=ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True


def _denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Normalized image tensor → HWC uint8 array suitable for imshow."""
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return (t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)


def _load_class_names(cfg: Config) -> dict[int, str]:
    """Returns {0-indexed class id → name} from objectInfo150.txt."""
    info_path = cfg.ade20k_root / "objectInfo150.txt"
    names = {}
    with open(info_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx = int(parts[0].strip())
            names[idx - 1] = parts[4].strip()  # convert to 0-indexed
    return names


def _draw_legend(ax: plt.Axes, mask_int: np.ndarray, class_names: dict[int, str]) -> None:
    """Draw color swatches, class names, and connected-component counts into ax."""
    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=149)

    present = sorted(c for c in np.unique(mask_int) if c != IGNORE_LABEL)
    entries = []
    for cls in present:
        _, n_obj = cc_label(mask_int == cls)
        entries.append((cmap(norm(cls)), class_names.get(cls, str(cls)), n_obj))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(entries))
    ax.axis("off")
    for i, (color, name, n_obj) in enumerate(reversed(entries)):
        y = i + 0.5
        ax.add_patch(plt.Rectangle((0, y - 0.35), 0.08, 0.7, color=color, transform=ax.transData))
        ax.text(0.12, y, f"{name}  ×{n_obj}", va="center", fontsize=7)


def show_val_samples(cfg: Config, n: int = 6, seed: int = 0, indices: list[int] | None = None):
    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    ds = ADE20kDataset(cfg.ade20k_root, "validation", img_transform=img_tf, mask_transform=mask_tf)
    class_names = _load_class_names(cfg)

    # indices are 1-based filename numbers (e.g. 55 → ADE_val_00000055.jpg, dataset index 54)
    if indices is not None:
        ds_indices = [i - 1 for i in indices]
    else:
        ds_indices = random.Random(seed).sample(range(len(ds)), n)

    fig, axes = plt.subplots(len(ds_indices), 3, figsize=(14, 4 * len(ds_indices)), squeeze=False)
    for row, idx in enumerate(ds_indices):
        img_t, mask_t = ds[idx]
        img = _denormalize(img_t)
        mask_int = mask_t.numpy()
        mask_display = mask_int.astype(float)
        mask_display[mask_int == IGNORE_LABEL] = np.nan

        name = ds.images[idx].stem
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(name)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_display, cmap="tab20", interpolation="nearest", vmin=0, vmax=149)
        axes[row, 1].set_title(f"{name}  (0–149, white = ignore)")
        axes[row, 1].axis("off")

        _draw_legend(axes[row, 2], mask_int, class_names)

    fig.suptitle(f"Validation — {cfg.resize_mode} {cfg.scene_size}×{cfg.scene_size}", fontsize=9)
    plt.tight_layout()
    return fig

# %%
if __name__ == "__main__":
    cfg = Config(probe_repo="canvit-probes", scene_size=512, resize_mode="squish")
    fig = show_val_samples(cfg, n=12, indices=[55, 395])
    plt.show()
