"""
Day 5 — GA-SVM Trainer
Uses the best chromosome from the GA run to train a final SVM,
then evaluates it against the held-out test set.
Produces the head-to-head comparison with Baseline SVM.
Run:  python backend/models/ga_svm_trainer.py
"""

import numpy as np
import json
import pickle
import time
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

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_all():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(RESULTS_DIR / "ga_results.json") as f:
        ga_results = json.load(f)
    with open(RESULTS_DIR / "baseline_results.json") as f:
        baseline = json.load(f)
    return X_train, X_test, y_train, y_test, meta, ga_results, baseline


def train_ga_svm(best_chr, X_train, y_train):
    mask  = np.array(best_chr["feature_mask"], dtype=bool)
    C     = best_chr["C"]
    gamma = best_chr["gamma"]

    X_sub = X_train[:, mask]
    clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42)

    t0 = time.perf_counter()
    clf.fit(X_sub, y_train)
    elapsed = time.perf_counter() - t0

    return clf, mask, elapsed


def evaluate(clf, mask, X_test, y_test):
    X_sub  = X_test[:, mask]
    y_pred = clf.predict(X_sub)
    y_prob = clf.predict_proba(X_sub)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n-- GA-SVM Evaluation ----------------------------------")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "accuracy":  round(float(acc), 4),
        "f1":        round(float(f1), 4),
        "auc":       round(float(auc), 4),
        "cm":        cm.tolist(),
        "y_prob":    y_prob.tolist(),
        "y_pred":    y_pred.tolist(),
        "roc":       {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def ten_fold_cv(mask, C, gamma, X_train, y_train, X_test, y_test):
    """Fix 4: 10-fold stratified CV on full dataset for final reported numbers."""
    from sklearn.model_selection import StratifiedKFold
    X_full = np.concatenate([X_train, X_test])[:, mask]
    y_full = np.concatenate([y_train, y_test])
    clf    = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42)
    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc_s, f1_s, auc_s = [], [], []
    for tr, vl in skf.split(X_full, y_full):
        clf.fit(X_full[tr], y_full[tr])
        yp = clf.predict(X_full[vl])
        yb = clf.predict_proba(X_full[vl])[:, 1]
        acc_s.append(accuracy_score(y_full[vl], yp))
        f1_s.append(f1_score(y_full[vl], yp, zero_division=0))
        auc_s.append(roc_auc_score(y_full[vl], yb))
    return {
        "cv10_accuracy_mean": round(float(np.mean(acc_s)), 4),
        "cv10_accuracy_std":  round(float(np.std(acc_s, ddof=1)), 4),
        "cv10_f1_mean":       round(float(np.mean(f1_s)), 4),
        "cv10_f1_std":        round(float(np.std(f1_s, ddof=1)), 4),
        "cv10_auc_mean":      round(float(np.mean(auc_s)), 4),
        "cv10_auc_std":       round(float(np.std(auc_s, ddof=1)), 4),
        "cv10_scores":        {"accuracy": acc_s, "f1": f1_s, "auc": auc_s},
    }
    """4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GA-SVM vs Baseline SVM — Comparison", fontsize=14, fontweight="bold")

    # 1. Metric bar chart
    ax = axes[0, 0]
    metrics  = ["Accuracy", "F1 Score", "AUC-ROC"]
    base_vals = [baseline["accuracy"], baseline["f1"], baseline["auc"]]
    gasvm_vals = [gasvm["accuracy"],    gasvm["f1"],    gasvm["auc"]]
    x = np.arange(len(metrics))
    w = 0.35
    bars_b = ax.bar(x - w/2, base_vals,  w, label="Baseline SVM", color="#4C72B0")
    bars_g = ax.bar(x + w/2, gasvm_vals, w, label="GA-SVM",       color="#DD8452")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Metric Comparison")
    ax.legend()
    for bar in [*bars_b, *bars_g]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # 2. Confusion matrix (GA-SVM)
    ax = axes[0, 1]
    sns.heatmap(np.array(gasvm["cm"]), annot=True, fmt="d", cmap="Oranges",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title("GA-SVM — Confusion Matrix")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

    # 3. ROC curves (both models)
    ax = axes[1, 0]
    b_roc = baseline["roc"]; g_roc = gasvm["roc"]
    ax.plot(b_roc["fpr"], b_roc["tpr"], lw=2,
            label=f"Baseline SVM (AUC={baseline['auc']:.3f})", color="#4C72B0")
    ax.plot(g_roc["fpr"], g_roc["tpr"], lw=2,
            label=f"GA-SVM (AUC={gasvm['auc']:.3f})", color="#DD8452")
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves"); ax.legend()

    # 4. Feature selection bar (which features GA chose)
    ax = axes[1, 1]
    colors = ["#DD8452" if m else "#CCCCCC" for m in mask]
    ax.barh(feature_names, mask.astype(int), color=colors)
    ax.set_xlim(0, 1.3)
    ax.set_title(f"GA Selected Features ({int(mask.sum())}/13)")
    ax.set_xlabel("Selected (1 = yes)")
    for i, v in enumerate(mask.astype(int)):
        ax.text(v + 0.02, i, "✓" if v else "✗", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Comparison saved → {out_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta, ga_results, baseline = load_all()
    best_chr = ga_results["best_chromosome"]
    feature_names = meta["feature_names"]

    # Train GA-SVM
    clf, mask, train_time = train_ga_svm(best_chr, X_train, y_train)
    results = evaluate(clf, mask, X_test, y_test)
    results["train_time_s"]   = round(train_time, 4)
    results["n_features_used"] = int(mask.sum())
    results["feature_mask"]   = mask.astype(int).tolist()
    results["C"]              = best_chr["C"]
    results["gamma"]          = best_chr["gamma"]
    results["ga_fitness"]     = best_chr["fitness"]

    # Fix 4: 10-fold CV for paper-quality numbers
    cv10 = ten_fold_cv(mask, best_chr["C"], best_chr["gamma"], X_train, y_train, X_test, y_test)
    results.update(cv10)
    print(f"\n[10-fold CV] Accuracy: {cv10['cv10_accuracy_mean']:.4f} +/- {cv10['cv10_accuracy_std']:.4f}")
    print(f"[10-fold CV] F1:       {cv10['cv10_f1_mean']:.4f} +/- {cv10['cv10_f1_std']:.4f}")
    print(f"[10-fold CV] AUC:      {cv10['cv10_auc_mean']:.4f} +/- {cv10['cv10_auc_std']:.4f}")

    # Side-by-side print
    print("\n── Head-to-head ────────────────────────────────────")
    print(f"{'Metric':<20} {'Baseline SVM':>14} {'GA-SVM':>14}")
    print("-" * 50)
    for metric in ["accuracy", "f1", "auc"]:
        print(f"  {metric:<18} {baseline[metric]:>14.4f} {results[metric]:>14.4f}")
    print(f"  {'features used':<18} {baseline['n_features_used']:>14} {results['n_features_used']:>14}")

    # Plots
    comparison_plot(
        baseline, results,
        feature_names, mask,
        RESULTS_DIR / "comparison.png"
    )

    # Save
    with open(RESULTS_DIR / "ga_svm_model.pkl", "wb") as f:
        pickle.dump({"model": clf, "mask": mask, "C": best_chr["C"], "gamma": best_chr["gamma"]}, f)
    with open(RESULTS_DIR / "ga_svm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Day 5] ✓  GA-SVM trained. Features: {int(mask.sum())}/13, "
          f"Accuracy: {results['accuracy']:.4f}, Time: {train_time:.4f}s")
