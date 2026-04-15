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
# ## Object Size Distributions

# %% Imports
import os

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.dataframes import Config, build_image_class_dataframe, class_stats_dataframe, _load_class_names

# %% Spatial distribution functions
def class_name_to_idx(name: str, cfg: Config) -> int:
    """Return the 1-based ADE20K class index for a class name.

    Matches against all comma-separated aliases in objectInfo150.txt (case-insensitive).
    Raises ValueError if the name doesn't match any of the 150 classes.
    """
    class_names = _load_class_names(cfg)
    name_lower = name.strip().lower()
    for idx, full_name in class_names.items():
        for part in full_name.split(","):
            if part.strip().lower() == name_lower:
                return idx
    raise ValueError(
        f"{name!r} does not match any ADE20K class. Check objectInfo150.txt for valid names."
    )


def plot_spatial_distribution(
    names: list[str],
    cfg: Config,
    batch_size: int = 64,
) -> matplotlib.figure.Figure:
    """Heatmap of spatial distribution for each named class across the validation set.

    Loads annotations in batches and computes mask sums with vectorised numpy operations.
    The result per pixel is P(pixel == class | class is present in the image).

    Args:
        names:      Class name strings (e.g. ["tree", "person"]). Raises ValueError for
                    any name that doesn't match an ADE20K class.
        cfg:        Config with ade20k_root and scene_size.
        batch_size: Number of annotation images to load at once.
    """
    class_indices = [class_name_to_idx(n, cfg) for n in names]
    size = cfg.scene_size
    n_classes = len(class_indices)

    # heatmaps[i] accumulates pixel-wise presence count for class_indices[i]
    heatmaps = np.zeros((n_classes, size, size), dtype=np.float64)
    appearances = np.zeros(n_classes, dtype=np.int64)

    ann_paths = sorted((cfg.ade20k_root / "annotations" / "validation").glob("*.png"))
    for batch_start in range(0, len(ann_paths), batch_size):
        batch = np.stack([
            np.array(Image.open(p).resize((size, size), Image.NEAREST))
            for p in ann_paths[batch_start : batch_start + batch_size]
        ])  # (B, size, size) uint8

        for i, idx in enumerate(class_indices):
            masks = batch == idx                        # (B, size, size) bool
            heatmaps[i] += masks.sum(axis=0)
            appearances[i] += masks.any(axis=(1, 2)).sum()

    for i in range(n_classes):
        if appearances[i] > 0:
            heatmaps[i] /= appearances[i]

    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (name, idx) in enumerate(zip(names, class_indices)):
        ax = axes_flat[i]
        im = ax.imshow(heatmaps[i], cmap="viridis", vmin=0, interpolation="nearest")
        ax.set_title(f"{name} (n={appearances[i]})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[len(names):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


# %% Plotting area functions
def plot_area_distribution(flat_df: pd.DataFrame, cfg: Config, n: int = 20, top: bool = True) -> matplotlib.figure.Figure:
    """Violin plot of per-image area fractions for the n largest or smallest classes by mean area."""
    class_names = _load_class_names(cfg)
    mean_area = flat_df.groupby("class_idx")["area"].mean()
    ranked = (mean_area.nlargest(n) if top else mean_area.nsmallest(n)).index.tolist()
    n_images = flat_df.groupby("class_idx").size()
    labels = [f"{class_names[cls].split(',')[0]}\n(n={n_images[cls]})" for cls in ranked]
    data = [flat_df.loc[flat_df["class_idx"] == cls, "area"].values for cls in ranked]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.violinplot(data, positions=range(len(data)), showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of image area")
    rank_label = "largest" if top else "smallest"
    ax.set_title(f"Area distribution — {n} {rank_label} classes by mean area (val set)")
    fig.tight_layout()
    return fig


def plot_metric_histogram(stats_df: pd.DataFrame, metric: str, top_n: int = 30) -> matplotlib.figure.Figure:
    """Bar chart of `metric` per class, showing the top_n classes sorted by that metric."""
    sorted_df = stats_df.nlargest(top_n, metric)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(sorted_df)), sorted_df[metric])
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df["name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — top {top_n} classes (val set)")
    fig.tight_layout()
    return fig


def filename_idx_to_img_idx(filename_idx: int) -> int:
    return filename_idx - 1


def img_idx_to_filename_idx(img_idx: int) -> int:
    return img_idx + 1

# %% Load dfs
if __name__ == "__main__":
    os.environ["ADE20K_ROOT"] = "../data/ADEChallengeData2016"
    cfg = Config(probe_repo="canvit-specialize")

    flat_df = build_image_class_dataframe(cfg)
    df = class_stats_dataframe(flat_df, cfg)

# %% Print dfs
if __name__ == "__main__":
    img_idx = filename_idx_to_img_idx(1809)
    print(flat_df.head())
    print(df.head(20))
    print(flat_df[flat_df["image_idx"] == img_idx])

# %% Show image and mask
if __name__ == "__main__":
    ann_dir = cfg.ade20k_root / "annotations" / "validation"
    img_dir = cfg.ade20k_root / "images" / "validation"

    ann_paths = sorted(ann_dir.glob("*.png"))
    img_paths = sorted(img_dir.glob("*.jpg"))

    ann = np.array(Image.open(ann_paths[img_idx]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(Image.open(img_paths[img_idx]))
    axes[0].set_title(f"Image (filename idx {img_idx_to_filename_idx(img_idx)})")
    axes[0].axis("off")

    axes[1].imshow(ann, cmap="tab20", interpolation="nearest", vmin=0, vmax=150)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# %% Run functions to show the histograms
if __name__ == "__main__":
    for metric in ("mean_area", "max_area"):
        plot_metric_histogram(df, metric)
        plt.show()

# %% Violins
if __name__ == "__main__":
    plot_area_distribution(flat_df, cfg, n=20, top=False)
    plt.show()

# %% Spatial heatmaps
if __name__ == "__main__":
    plot_spatial_distribution(names=["sky", "building", "person"], cfg=cfg)
    plt.show()
