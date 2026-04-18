"""
Data Pipeline
Downloads and merges all 4 UCI Heart Disease sub-datasets
(Cleveland, Hungarian, Switzerland, VA Long Beach), cleans,
scales, and saves train/test splits.

Combined dataset: ~900 patients vs 303 (Cleveland only).
Run:  python prepare_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json

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
    "age":      {"min": 28,  "max": 77,   "step": 1,   "type": "int"},
    "sex":      {"min": 0,   "max": 1,    "step": 1,   "type": "int"},
    "cp":       {"min": 0,   "max": 3,    "step": 1,   "type": "int"},
    "trestbps": {"min": 0,   "max": 200,  "step": 1,   "type": "int"},
    "chol":     {"min": 0,   "max": 603,  "step": 1,   "type": "int"},
    "fbs":      {"min": 0,   "max": 1,    "step": 1,   "type": "int"},
    "restecg":  {"min": 0,   "max": 2,    "step": 1,   "type": "int"},
    "thalach":  {"min": 60,  "max": 202,  "step": 1,   "type": "int"},
    "exang":    {"min": 0,   "max": 1,    "step": 1,   "type": "int"},
    "oldpeak":  {"min": 0.0, "max": 6.2,  "step": 0.1, "type": "float"},
    "slope":    {"min": 0,   "max": 2,    "step": 1,   "type": "int"},
    "ca":       {"min": 0,   "max": 4,    "step": 1,   "type": "int"},
    "thal":     {"min": 0,   "max": 3,    "step": 1,   "type": "int"},
}

# All 4 UCI sub-datasets — same 13 features, same format
UCI_URLS = {
    "cleveland":  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian":  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland":"https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va":         "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

DATA_DIR = Path(__file__).parent / "data"


def download_all() -> pd.DataFrame:
    frames = []
    for name, url in UCI_URLS.items():
        try:
            df = pd.read_csv(url, names=COLUMNS, na_values="?")
            df["source"] = name
            print(f"[data] {name:12s} — {len(df)} rows, "
                  f"missing: ca={df['ca'].isnull().sum()} thal={df['thal'].isnull().sum()}")
            frames.append(df)
        except Exception as e:
            print(f"[data] {name} download failed: {e}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[data] Combined: {len(combined)} rows from {len(frames)} datasets")
    return combined


def inspect(df: pd.DataFrame) -> None:
    print("\n-- Shape ----------------------------------")
    print(df.shape)
    print("\n-- Missing values -------------------------")
    print(df.isnull().sum())
    print("\n-- Class distribution (raw) ---------------")
    print(df["target"].value_counts().sort_index())
    print("\n-- Source breakdown -----------------------")
    print(df.groupby("source")["target"].apply(lambda x: (x > 0).sum()).rename("disease")
          .to_frame().assign(total=df.groupby("source").size()))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binarize target: 0 = no disease, 1 = disease (original values 1-4)
    df["target"] = (df["target"] > 0).astype(int)

    # Drop source column before modelling
    df = df.drop(columns=["source"], errors="ignore")

    # Mode-impute missing values per column across the full combined dataset
    for col in COLUMNS[:-1]:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[clean] Imputed {missing:3d} missing in '{col}' with mode={mode_val}")

    df = df.astype(float)
    return df


def preprocess(df: pd.DataFrame):
    feature_cols = COLUMNS[:-1]
    X = df[feature_cols].values
    y = df["target"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\n[preprocess] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[preprocess] Class balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    print(f"[preprocess] Class balance (test)  — 0: {(y_test==0).sum()},  1: {(y_test==1).sum()}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_cols, n_total, source_counts):
    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "X_train.npy", X_train)
    np.save(DATA_DIR / "X_test.npy",  X_test)
    np.save(DATA_DIR / "y_train.npy", y_train)
    np.save(DATA_DIR / "y_test.npy",  y_test)

    with open(DATA_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_names":        feature_cols,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "feature_ranges":       FEATURE_RANGES,
        "n_features":           len(feature_cols),
        "n_total":              n_total,
        "n_train":              len(y_train),
        "n_test":               len(y_test),
        "sources":              source_counts,
        "class_balance": {
            "train": {"0": int((y_train==0).sum()), "1": int((y_train==1).sum())},
            "test":  {"0": int((y_test==0).sum()),  "1": int((y_test==1).sum())}
        }
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[save] Artifacts saved to {DATA_DIR}")


if __name__ == "__main__":
    df_raw   = download_all()
    inspect(df_raw)

    source_counts = df_raw.groupby("source").size().to_dict()

    df_clean = clean(df_raw)
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df_clean)
    save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_cols,
                   n_total=len(df_clean), source_counts=source_counts)

    print(f"\n[OK] Data pipeline complete. {len(df_clean)} total patients.")
