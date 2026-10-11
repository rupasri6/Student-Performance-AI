"""Train/test split helpers and a sklearn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_BASE,
    ENGINEERED_NUMERIC,
    FAIRNESS_EXCLUDED,
    LEAKY_FEATURES,
    NUMERIC_BASE,
    TARGET_CLASS,
    TARGET_REGRESSION,
)
from src.data.features import add_support_label, engineer_features


def prepare_modelling_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = engineer_features(raw)
    frame = add_support_label(frame)
    drop_cols = [c for c in LEAKY_FEATURES + FAIRNESS_EXCLUDED if c in frame.columns]
    return frame.drop(columns=drop_cols)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    missing = [c for c in ALL_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Prepared frame is missing features: {missing}")
    X = frame[ALL_FEATURES]
    y_reg = frame[TARGET_REGRESSION]
    y_cls = frame[TARGET_CLASS]
    return X, y_reg, y_cls


def build_preprocessor() -> ColumnTransformer:
    numeric = NUMERIC_BASE + ENGINEERED_NUMERIC
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, CATEGORICAL_BASE),
        ]
    )
