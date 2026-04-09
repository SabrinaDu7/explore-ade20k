"""ADE20K segmentation eval — model-agnostic config and root resolution."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ResizeMode = Literal["center_crop", "squish"]


def ade20k_root() -> Path:
    if root := os.environ.get("ADE20K_ROOT"):
        return Path(root)
    if tmpdir := os.environ.get("SLURM_TMPDIR"):
        return Path(tmpdir) / "ADEChallengeData2016"
    raise ValueError("ADE20K_ROOT env var not set and not running under SLURM")


@dataclass
class Config:
    """ADE20K segmentation eval — model-agnostic."""

    probe_repo: str
    ade20k_root: Path = field(default_factory=ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True
