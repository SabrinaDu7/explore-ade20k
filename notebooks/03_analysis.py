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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label as cc_label

from src.dataset import ade20k_root

ResizeMode = Literal["center_crop", "squish"]

# %% Computing functions
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


def _load_class_names(cfg: Config) -> dict[int, str]:
    info_path = cfg.ade20k_root / "objectInfo150.txt"
    names = {}
    with open(info_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx = int(parts[0].strip())
            names[idx] = parts[4].strip()
    return names


def compute_class_area(ann: np.ndarray, class_idx: int) -> float:
    """Fraction of pixels in ann occupied by class_idx (1-indexed)."""
    return float((ann == class_idx).sum() / ann.size)


def build_image_class_dataframe(cfg: Config) -> pd.DataFrame:
    """One row per (image, class) pair observed in the validation set.

    Columns:
        image_idx   int   0-based index into the sorted annotation file list
        class_idx   int   1-based ADE20K class index
        class_name  str   primary name of the class (first comma-separated entry)
        area        float fraction of image pixels occupied by the class
        object_count int  number of connected components for the class in this image
    """
    ann_dir = cfg.ade20k_root / "annotations" / "validation"
    class_names = _load_class_names(cfg)
    rows = []
    for image_idx, ann_path in enumerate(sorted(ann_dir.glob("*.png"))):
        arr = np.array(Image.open(ann_path))
        for cls in np.unique(arr).tolist():
            if cls == 0 or cls > 150:
                continue
            mask = arr == cls
            area = compute_class_area(arr, cls)
            n_obj = int(cc_label(mask)[1])
            rows.append({
                "image_idx": image_idx,
                "class_idx": cls,
                "class_name": class_names[cls].split(",")[0],
                "area": area,
                "object_count": n_obj,
            })
    return pd.DataFrame(rows)


def class_area_range(flat_df: pd.DataFrame, class_idx: int) -> tuple[float, float]:
    """Return (min_area, max_area) across all validation images for class_idx."""
    areas = flat_df.loc[flat_df["class_idx"] == class_idx, "area"]
    if areas.empty:
        return (0.0, 0.0)
    return float(areas.min()), float(areas.max())


def compute_object_count(flat_df: pd.DataFrame, class_idx: int) -> int:
    """Total connected-component count for class_idx across the validation set."""
    return int(flat_df.loc[flat_df["class_idx"] == class_idx, "object_count"].sum())


def class_stats_dataframe(flat_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate flat_df to one row per class: object_count, mean_area, max_area, min_area."""
    class_names = _load_class_names(cfg)
    stats = flat_df.groupby("class_idx").agg(
        object_count=("object_count", "sum"),
        mean_area=("area", "mean"),
        max_area=("area", "max"),
        min_area=("area", "min"),
    )
    stats["name"] = stats.index.map(lambda i: class_names[i].split(",")[0])
    return stats.sort_values("object_count", ascending=False)


# %% Plotting functions
def plot_area_distribution(flat_df: pd.DataFrame, cfg: Config, n: int = 20, top: bool = True) -> matplotlib.figure.Figure:
    """Violin plot of per-image area fractions for the top or bottom n most frequent classes."""
    class_names = _load_class_names(cfg)
    counts = flat_df.groupby("class_idx").size()
    ranked = (counts.nlargest(n) if top else counts.nsmallest(n)).index.tolist()
    labels = [class_names[cls].split(",")[0] for cls in ranked]
    data = [flat_df.loc[flat_df["class_idx"] == cls, "area"].values for cls in ranked]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.violinplot(data, positions=range(len(data)), showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of image area")
    rank_label = "top" if top else "bottom"
    ax.set_title(f"Area distribution — {rank_label} {n} classes by image count (val set)")
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


def plot_size_vs_instances_violin(flat_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Violin plot of object_count distributions binned by class area (% of image).

    X-axis: area bins (log-spaced to handle the heavy skew toward small objects)
    Y-axis: object_count (connected components per image-class pair)
    """
    pct = flat_df["area"] * 100
    bin_edges = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    bin_labels = ["0–0.1", "0.1–0.5", "0.5–1", "1–2", "2–5", "5–10", "10–20", "20–50", "50–100"]
    bins = pd.cut(pct, bins=bin_edges, labels=bin_labels, right=True)
    grouped = flat_df.groupby(bins, observed=True)["object_count"]
    data = [g.values for _, g in grouped if len(g) > 1]
    labels = [lbl for lbl, g in grouped if len(g) > 1]
    counts = [len(g) for _, g in grouped if len(g) > 1]

    fig, ax = plt.subplots(figsize=(13, 5))
    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [f"{lbl}%\n(n={c})" for lbl, c in zip(labels, counts)],
        fontsize=8,
    )
    ax.set_xlabel("Object class area (% of image)")
    ax.set_ylabel("Instances per image (connected components)")
    ax.set_title("Instance count distribution by object size bin (flat_df, val set)")
    fig.tight_layout()
    return fig


def filename_idx_to_img_idx(filename_idx: int) -> int:
    return filename_idx - 1


def img_idx_to_filename_idx(img_idx: int) -> int:
    return img_idx + 1


# %% Run functions to show the dataframes
if __name__ == "__main__":
    os.environ["ADE20K_ROOT"] = "../data/ADEChallengeData2016/"
    cfg = Config(probe_repo="canvit-probes")
    flat_df = build_image_class_dataframe(cfg)
    # print(flat_df.head())
    df = class_stats_dataframe(flat_df, cfg)
    print(df.head(20))
    img_idx = filename_idx_to_img_idx(1809)
    print(flat_df[flat_df["image_idx"] == img_idx])

# %% Run functions to show the histograms
if __name__ == "__main__":
    plot_area_distribution(flat_df, cfg, n=20, top=False)
    plt.show()
    for metric in ("object_count", "mean_area", "max_area"):
        plot_metric_histogram(df, metric)
        plt.show()

# %% Size vs instances violin
if __name__ == "__main__":
    plot_size_vs_instances_violin(flat_df)
    plt.show()

