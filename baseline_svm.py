"""
Day 2 — Baseline SVM
Trains a plain SVM on all 13 features with default hyperparameters.
Records metrics used later for comparison against GA-SVM.
Run:  python backend/models/baseline_svm.py
"""

import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score
import time

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    return X_train, X_test, y_train, y_test, meta


def train_baseline(X_train, y_train):
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return clf, train_time


def evaluate(clf, X_test, y_test, label="Baseline SVM"):
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n── {label} ──────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return {
        "label":    label,
        "accuracy": round(float(acc), 4),
        "f1":       round(float(f1), 4),
        "auc":      round(float(auc), 4),
        "cm":       cm.tolist(),
        "report":   report,
        "y_prob":   y_prob.tolist(),
        "y_pred":   y_pred.tolist(),
    }


def plot_confusion_matrix(cm, label, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"], ax=ax
    )
    ax.set_title(f"Confusion Matrix — {label}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Saved confusion matrix → {path}")


def plot_roc(fpr, tpr, auc_val, label, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] Saved ROC curve → {path}")


def cross_validate(clf_params, X_train, y_train, cv=5):
    clf = SVC(**clf_params, probability=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    print(f"\n[CV] {cv}-fold F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.tolist()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = load_data()

    # 5-fold CV on training set
    cv_scores = cross_validate(
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        X_train, y_train
    )

    # Train and evaluate
    clf, train_time = train_baseline(X_train, y_train)
    results = evaluate(clf, X_test, y_test, "Baseline SVM (all 13 features)")
    results["train_time_s"] = round(train_time, 4)
    results["cv_f1_scores"] = cv_scores
    results["n_features_used"] = 13
    results["feature_mask"] = [1] * 13

    # ROC data
    y_prob = np.array(results["y_prob"])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    results["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Plots
    plot_confusion_matrix(
        np.array(results["cm"]), "Baseline SVM",
        RESULTS_DIR / "baseline_cm.png"
    )
    plot_roc(fpr, tpr, results["auc"], "Baseline SVM",
             RESULTS_DIR / "baseline_roc.png")

    # Save model and results
    with open(RESULTS_DIR / "baseline_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Day 2] ✓  Baseline SVM complete. Train time: {train_time:.4f}s")
