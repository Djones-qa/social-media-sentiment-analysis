"""
utils.py — Shared utilities: config, formatting, diagnostics.
"""

import os
import yaml
import pandas as pd
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_number(value):
    if pd.isna(value):
        return "N/A"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


def format_pct(value, decimals=1):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def dataset_summary(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicates": df.duplicated().sum(),
    }


def print_summary(df, label="Dataset"):
    info = dataset_summary(df)
    print(f"\n{'='*50}")
    print(f"  {label} Summary")
    print(f"{'='*50}")
    print(f"  Rows:        {info['rows']:,}")
    print(f"  Columns:     {info['columns']}")
    print(f"  Missing:     {info['missing_pct']}%")
    print(f"  Duplicates:  {info['duplicates']:,}")
    print(f"  Memory:      {info['memory_mb']} MB")
    print(f"{'='*50}\n")


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
