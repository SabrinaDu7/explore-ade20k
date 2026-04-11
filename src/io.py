"""Persistence utilities for ADE20K analysis outputs."""

import os
from pathlib import Path
import pandas as pd

def ade20k_root() -> Path:
    if root := os.environ.get("ADE20K_ROOT"):
        return Path(root)
    if tmpdir := os.environ.get("SLURM_TMPDIR"):
        return Path(tmpdir) / "ADEChallengeData2016"
    raise ValueError("ADE20K_ROOT env var not set and not running under SLURM")
