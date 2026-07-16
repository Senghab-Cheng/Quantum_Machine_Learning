"""Reusable preprocessing pipeline for single-sample inference.

`preprocessing.preprocess()` fits a fresh StandardScaler/PCA on every call
and discards them, which works for one-off train/test splits but can't
transform a single new patient consistently with how the training data was
transformed. This module fits those transformers once and keeps them, so a
new raw patient record can be mapped into the same quantum-encoded feature
space the models were trained on.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config
from data_loader import load_data

RAW_FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_IS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def fit_pipeline() -> Dict[str, Any]:
    """Load the dataset and fit the full preprocessing pipeline once.

    Returns a dict holding the fitted transformers plus the same
    train/test split `preprocessing.preprocess()` produces, so results
    stay consistent with the rest of the project.
    """
    X, y = load_data()
    df = X.copy()
    df["Outcome"] = y

    X_raw = df[RAW_FEATURE_COLUMNS].copy()
    y_all = df["Outcome"].copy()

    medians = {}
    for col in ZERO_IS_MISSING_COLUMNS:
        median_val = X_raw[col][X_raw[col] > 0].median()
        medians[col] = median_val
        X_raw.loc[X_raw[col] == 0, col] = median_val

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    scale_min = X_scaled.min(axis=0)
    scale_max = X_scaled.max(axis=0)
    X_scaled_norm = (X_scaled - scale_min) / (scale_max - scale_min + 1e-8)

    pca = PCA(n_components=config.N_QUBITS)
    X_pca = pca.fit_transform(X_scaled_norm)

    pca_min = X_pca.min(axis=0)
    pca_max = X_pca.max(axis=0)
    X_pca_norm = (X_pca - pca_min) / (pca_max - pca_min + 1e-8)
    X_quantum = X_pca_norm * np.pi

    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y_all,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_all,
    )

    return {
        "medians": medians,
        "scaler": scaler,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "pca": pca,
        "pca_min": pca_min,
        "pca_max": pca_max,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "raw_df": df,
    }


def transform_sample(raw_features: Dict[str, float], pipeline: Dict[str, Any]) -> np.ndarray:
    """Map a single raw patient record through the fitted pipeline.

    `raw_features` must have one entry per `RAW_FEATURE_COLUMNS`. Returns a
    (1, N_QUBITS) array in the same [0, pi] quantum-encoded space as
    `pipeline["X_train"]`.
    """
    row = pd.DataFrame([raw_features], columns=RAW_FEATURE_COLUMNS)

    for col in ZERO_IS_MISSING_COLUMNS:
        if row.at[0, col] == 0:
            row.at[0, col] = pipeline["medians"][col]

    scaled = pipeline["scaler"].transform(row)
    scaled_norm = (scaled - pipeline["scale_min"]) / (pipeline["scale_max"] - pipeline["scale_min"] + 1e-8)

    pca_out = pipeline["pca"].transform(scaled_norm)
    pca_norm = (pca_out - pipeline["pca_min"]) / (pipeline["pca_max"] - pipeline["pca_min"] + 1e-8)
    quantum = np.clip(pca_norm, 0.0, 1.0) * np.pi

    return quantum
