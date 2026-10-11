"""Load and combine the UCI Student Performance Math and Portuguese files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MATH_CSV, NUMERIC_BASE, POR_CSV

NUMERIC_ALWAYS = list(NUMERIC_BASE) + ["G1", "G2", "G3"]


def load_raw_dataset(math_path=MATH_CSV, por_path=POR_CSV) -> pd.DataFrame:
    """Return one table with a `subject` column. Duplicate course rows are kept.

    A student who takes both Math and Portuguese is two academic observations,
    not a data-quality duplicate.
    """
    math_df = _read_uci_csv(math_path)
    por_df = _read_uci_csv(por_path)
    math_df["subject"] = "math"
    por_df["subject"] = "portuguese"
    frame = pd.concat([math_df, por_df], ignore_index=True)
    frame.columns = [c.strip() for c in frame.columns]
    for col in NUMERIC_ALWAYS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _read_uci_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {path}. Place student-mat.csv and student-por.csv in data/raw "
            "or run: python -m src.data.download"
        )
    return pd.read_csv(path, sep=";")
