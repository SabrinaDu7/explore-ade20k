"""Persistence utilities for ADE20K analysis outputs."""

from pathlib import Path

import pandas as pd


def save_dataframes(flat_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Save flat_df and stats_df to parquet in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_df.to_parquet(output_dir / "ade20k_df_flat.parquet")
    stats_df.to_parquet(output_dir / "ade20k_df_stats.parquet")
