"""
Day 1 — Data Pipeline
Downloads the UCI Cleveland Heart Disease dataset, inspects it,
handles missing values, encodes categoricals, and normalises features.
Run:  python backend/data/prepare_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json

# ── Column names per UCI specification ──────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

FEATURE_DESCRIPTIONS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=male, 0=female)",
    "cp":       "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol":     "Serum cholesterol (mg/dl)",
    "fbs":      "Fasting blood sugar > 120 (1=true)",
    "restecg":  "Resting ECG results (0-2)",
    "thalach":  "Max heart rate achieved",
    "exang":    "Exercise-induced angina (1=yes)",
    "oldpeak":  "ST depression (exercise vs rest)",
    "slope":    "Slope of peak exercise ST segment",
    "ca":       "Major vessels coloured by fluoroscopy",
    "thal":     "Thalassemia (1=normal, 2=fixed, 3=reversible defect)"
}

FEATURE_RANGES = {
    "age":      {"min": 29,  "max": 77,   "step": 1,   "type": "int"},
    "sex":      {"min": 0,   "max": 1,    "step": 1,   "type": "int"},
    "cp":       {"min": 0,   "max": 3,    "step": 1,   "type": "int"},
    "trestbps": {"min": 94,  "max": 200,  "step": 1,   "type": "int"},
    "chol":     {"min": 126, "max": 564,  "step": 1,   "type": "int"},
    "fbs":      {"min": 0,   "max": 1,    "step": 1,   "type": "int"},
    "restecg":  {"min": 0,   "max": 2,    "step": 1,   "type": "int"},
    "thalach":  {"min": 71,  "max": 202,  "step": 1,   "type": "int"},
    "exang":    {"min": 0,   "max": 1,    "step": 1,   "type": "int"},
    "oldpeak":  {"min": 0.0, "max": 6.2,  "step": 0.1, "type": "float"},
    "slope":    {"min": 0,   "max": 2,    "step": 1,   "type": "int"},
    "ca":       {"min": 0,   "max": 4,    "step": 1,   "type": "int"},
    "thal":     {"min": 0,   "max": 3,    "step": 1,   "type": "int"},
}

DATA_DIR = Path(__file__).parent


def download_dataset() -> pd.DataFrame:
    """
    Fetches the Cleveland heart disease dataset from UCI ML Repository.
    Falls back to a bundled minimal CSV if network is unavailable.
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    try:
        df = pd.read_csv(url, names=COLUMNS, na_values="?")
        print(f"[data] Downloaded {len(df)} rows from UCI repository.")
    except Exception as e:
        print(f"[data] Download failed ({e}). Using local fallback.")
        df = pd.read_csv(DATA_DIR / "heart.csv", names=COLUMNS, na_values="?")
    return df


def inspect(df: pd.DataFrame) -> None:
    print("\n── Shape ──────────────────────────")
    print(df.shape)
    print("\n── Dtypes ─────────────────────────")
    print(df.dtypes)
    print("\n── Missing values ─────────────────")
    print(df.isnull().sum())
    print("\n── Class distribution ─────────────")
    print(df["target"].value_counts())
    print("\n── Descriptive stats ──────────────")
    print(df.describe())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Binarize target (0 = no disease, 1 = disease)
    2. Mode-impute the two columns with '?' (ca, thal)
    3. Cast to float
    """
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)

    for col in ["ca", "thal"]:
        mode_val = df[col].mode()[0]
        missing = df[col].isnull().sum()
        df[col] = df[col].fillna(mode_val)
        print(f"[clean] Imputed {missing} missing values in '{col}' with mode={mode_val}")

    df = df.astype(float)
    return df


def preprocess(df: pd.DataFrame):
    """
    Returns:
        X_train, X_test, y_train, y_test  (numpy arrays)
        scaler                             (fitted StandardScaler)
        feature_names                      (list of 13 strings)
    """
    feature_cols = COLUMNS[:-1]   # everything except 'target'
    X = df[feature_cols].values
    y = df["target"].values.astype(int)

    # Stratified split — preserves class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n[preprocess] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[preprocess] Class balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_cols):
    out = DATA_DIR
    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_test.npy",  X_test)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "y_test.npy",  y_test)

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_names":       feature_cols,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "feature_ranges":      FEATURE_RANGES,
        "n_features":          len(feature_cols),
        "n_train":             len(y_train),
        "n_test":              len(y_test),
        "class_balance": {
            "train": {"0": int((y_train==0).sum()), "1": int((y_train==1).sum())},
            "test":  {"0": int((y_test==0).sum()),  "1": int((y_test==1).sum())}
        }
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[save] Artifacts saved to {out}")


if __name__ == "__main__":
    df_raw  = download_dataset()
    inspect(df_raw)
    df_clean = clean(df_raw)
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df_clean)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_cols)
    print("\n[Day 1] ✓  Data pipeline complete.")