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
# # Dataset Overview
# High-level stats: label distribution, resolution distribution, aspect ratios.

# %%
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os


import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.io import ade20k_root

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

# %% [markdown]
# ## LABELS

# %%
def _load_class_names(cfg: Config) -> dict[int, str]:
    info_path = cfg.ade20k_root / "objectInfo150.txt"
    names = {}
    with open(info_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx = int(parts[0].strip())
            name = parts[4].strip()
            names[idx] = name
    return names


def absent_classes(cfg: Config) -> list[str]:
    class_names = _load_class_names(cfg)
    ann_dir = cfg.ade20k_root / "annotations" / "validation"
    seen: set[int] = set()
    for ann_path in sorted(ann_dir.glob("*.png")):
        arr = np.array(Image.open(ann_path))
        seen.update(np.unique(arr).tolist())
    seen.discard(0)  # 0 = unlabeled/ignore

    absent = [class_names[i] for i in range(1, 151) if i not in seen]
    print(f"Absent classes ({len(absent)}):", absent if absent else "none")
    return absent


def plot_label_distribution(cfg: Config, top_n: int = 30) -> matplotlib.figure.Figure:
    class_names = _load_class_names(cfg)
    ann_dir = cfg.ade20k_root / "annotations" / "validation"
    counts = np.zeros(151, dtype=np.int64)  # 1-indexed, index 0 unused
    for ann_path in sorted(ann_dir.glob("*.png")):
        arr = np.array(Image.open(ann_path))
        unique, freq = np.unique(arr, return_counts=True)
        for label, cnt in zip(unique.tolist(), freq.tolist()):
            if 1 <= label <= 150:
                counts[label] += cnt

    present_indices = np.where(counts[1:] > 0)[0] + 1
    present_counts = counts[present_indices]
    order = np.argsort(present_counts)[::-1][:top_n]
    top_indices = present_indices[order]
    top_counts = present_counts[order]
    short_names = [class_names[i].split(",")[0] for i in top_indices.tolist()]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(top_counts)), top_counts)
    ax.set_yscale("log")
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pixel count (log scale)")
    ax.set_title(f"Val set — top {top_n} classes by pixel frequency")
    fig.tight_layout()
    return fig

# %%
if __name__ == "__main__":
    os.environ["ADE20K_ROOT"] = "../data/ADEChallengeData2016/"
    cfg = Config(probe_repo="canvit-probes")
    absent_classes(cfg)
    plot_label_distribution(cfg, top_n=30)
    plt.show()

# %% [markdown]
# ## IMAGE INFO

# %%
NAMED_RATIOS = {"1:1": 1/1, "3:2": 3/2, "4:3": 4/3, "16:9": 16/9}
TOLERANCE = 0.05


def _classify_ratio(w: int, h: int) -> str:
    ratio = w / h
    for name, target in NAMED_RATIOS.items():
        if abs(ratio - target) / target <= TOLERANCE:
            return name
        if target != 1 and abs(ratio - (1 / target)) / (1 / target) <= TOLERANCE:
            return f"{name} portrait"
    return "other"


def _collect_sizes(cfg: Config) -> list[tuple[int, int]]:
    img_dir = cfg.ade20k_root / "images" / "validation"
    sizes = []
    for path in sorted(img_dir.glob("*.jpg")):
        with Image.open(path) as im:
            sizes.append(im.size)
    return sizes


def plot_resolution_distribution(cfg: Config) -> matplotlib.figure.Figure:
    sizes = _collect_sizes(cfg)
    widths = np.array([s[0] for s in sizes])
    heights = np.array([s[1] for s in sizes])

    fig, ax = plt.subplots(figsize=(7, 6))
    h, xedges, yedges = np.histogram2d(widths, heights, bins=40)
    ax.pcolormesh(xedges, yedges, h.T, cmap="viridis")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=h.min(), vmax=h.max())),
        ax=ax, label="Count",
    )
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Resolution distribution (val set)")
    fig.tight_layout()
    return fig


def plot_aspect_ratio_distribution(cfg: Config) -> matplotlib.figure.Figure:
    sizes = _collect_sizes(cfg)
    bucket_order = list(NAMED_RATIOS) + [f"{k} portrait" for k in NAMED_RATIOS if k != "1:1"] + ["other"]
    counts: dict[str, int] = {b: 0 for b in bucket_order}
    for w, h in sizes:
        counts[_classify_ratio(w, h)] += 1

    labels = [b for b in bucket_order if counts[b] > 0]
    values = [counts[b] for b in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_xlabel("Aspect ratio bucket")
    ax.set_ylabel("Count")
    ax.set_title("Aspect ratio distribution (val set)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_resolution_vs_aspect(cfg: Config) -> matplotlib.figure.Figure:
    sizes = _collect_sizes(cfg)
    bucket_order = list(NAMED_RATIOS) + [f"{k} portrait" for k in NAMED_RATIOS if k != "1:1"] + ["other"]
    palette = plt.cm.get_cmap("tab10", len(bucket_order))
    color_map = {b: palette(i) for i, b in enumerate(bucket_order)}

    grouped: dict[str, list[tuple[int, int]]] = {b: [] for b in bucket_order}
    for w, h in sizes:
        grouped[_classify_ratio(w, h)].append((w, h))

    fig, ax = plt.subplots(figsize=(7, 6))
    for bucket, pts in grouped.items():
        if not pts:
            continue
        ws, hs = zip(*pts)
        ax.scatter(ws, hs, s=8, alpha=0.6, color=color_map[bucket], label=bucket)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Width vs height by aspect ratio (val set)")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    return fig

# %%
if __name__ == "__main__":
    cfg = Config(probe_repo="canvit-probes")
    plot_resolution_distribution(cfg)
    plt.show()
    plot_aspect_ratio_distribution(cfg)
    plt.show()
    plot_resolution_vs_aspect(cfg)
    plt.show()
