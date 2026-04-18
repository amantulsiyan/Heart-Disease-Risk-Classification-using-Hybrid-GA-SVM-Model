"""
Statistical Analysis
Fix 1: 5-seed repeated evaluation with mean +/- std
Fix 4: 10-fold stratified cross-validation for final reported numbers
Wilcoxon signed-rank test comparing GA-SVM and GA-MLP vs Baseline.

Run AFTER all training scripts have completed:
    python statistical_analysis.py
"""

import numpy as np
import json
import pickle
import time
import logging
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SEEDS       = [42, 7, 13, 99, 2024]
CV_FOLDS    = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    X = np.concatenate([
        np.load(DATA_DIR / "X_train.npy"),
        np.load(DATA_DIR / "X_test.npy")
    ])
    y = np.concatenate([
        np.load(DATA_DIR / "y_train.npy"),
        np.load(DATA_DIR / "y_test.npy")
    ])
    return X, y


def ci95(scores):
    """95% confidence interval assuming normal distribution."""
    mean = np.mean(scores)
    std  = np.std(scores, ddof=1)
    n    = len(scores)
    margin = 1.96 * std / np.sqrt(n)
    return mean, std, mean - margin, mean + margin


def eval_fold(clf, X_tr, y_tr, X_vl, y_vl):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_vl)
    y_prob = clf.predict_proba(X_vl)[:, 1]
    return {
        "accuracy": accuracy_score(y_vl, y_pred),
        "f1":       f1_score(y_vl, y_pred, zero_division=0),
        "auc":      roc_auc_score(y_vl, y_prob),
    }


# ── Fix 4: 10-fold CV evaluation ──────────────────────────────────────────────

def ten_fold_cv_baseline(X, y):
    """10-fold CV for baseline SVM (all 13 features, default params)."""
    log.info("[10-fold] Evaluating Baseline SVM...")
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    acc_scores, f1_scores, auc_scores = [], [], []

    for tr, vl in skf.split(X, y):
        res = eval_fold(clf, X[tr], y[tr], X[vl], y[vl])
        acc_scores.append(res["accuracy"])
        f1_scores.append(res["f1"])
        auc_scores.append(res["auc"])

    return {
        "accuracy": acc_scores,
        "f1":       f1_scores,
        "auc":      auc_scores,
    }


def ten_fold_cv_gasvm(X, y):
    """10-fold CV for GA-SVM using the best chromosome found by the GA."""
    log.info("[10-fold] Evaluating GA-SVM...")
    with open(RESULTS_DIR / "ga_results.json") as f:
        ga_res = json.load(f)
    best = ga_res["best_chromosome"]
    mask  = np.array(best["feature_mask"], dtype=bool)
    C     = best["C"]
    gamma = best["gamma"]

    clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    acc_scores, f1_scores, auc_scores = [], [], []

    for tr, vl in skf.split(X[:, mask], y):
        res = eval_fold(clf, X[tr][:, mask], y[tr], X[vl][:, mask], y[vl])
        acc_scores.append(res["accuracy"])
        f1_scores.append(res["f1"])
        auc_scores.append(res["auc"])

    return {
        "accuracy": acc_scores,
        "f1":       f1_scores,
        "auc":      auc_scores,
        "n_features": int(mask.sum()),
        "C": C,
        "gamma": gamma,
    }


# ── Fix 1: 5-seed repeated evaluation ────────────────────────────────────────

def repeated_eval_baseline(X, y):
    """Run baseline SVM with 5 different random seeds, report mean +/- std."""
    log.info("[5-seed] Repeated evaluation: Baseline SVM...")
    all_acc, all_f1, all_auc = [], [], []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
        for tr, vl in skf.split(X, y):
            res = eval_fold(clf, X[tr], y[tr], X[vl], y[vl])
            all_acc.append(res["accuracy"])
            all_f1.append(res["f1"])
            all_auc.append(res["auc"])

    return {"accuracy": all_acc, "f1": all_f1, "auc": all_auc}


def repeated_eval_gasvm(X, y):
    """Run GA-SVM (fixed best chromosome) with 5 seeds."""
    log.info("[5-seed] Repeated evaluation: GA-SVM...")
    with open(RESULTS_DIR / "ga_results.json") as f:
        ga_res = json.load(f)
    best  = ga_res["best_chromosome"]
    mask  = np.array(best["feature_mask"], dtype=bool)
    C     = best["C"]
    gamma = best["gamma"]
    X_sub = X[:, mask]

    all_acc, all_f1, all_auc = [], [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed)
        for tr, vl in skf.split(X_sub, y):
            res = eval_fold(clf, X_sub[tr], y[tr], X_sub[vl], y[vl])
            all_acc.append(res["accuracy"])
            all_f1.append(res["f1"])
            all_auc.append(res["auc"])

    return {"accuracy": all_acc, "f1": all_f1, "auc": all_auc}


# ── Fix 1: Wilcoxon signed-rank test ─────────────────────────────────────────

def significance_tests(baseline_scores, gasvm_scores, label="GA-SVM vs Baseline"):
    """Paired Wilcoxon signed-rank test + paired t-test on per-fold scores."""
    results = {}
    for metric in ["accuracy", "f1", "auc"]:
        b = np.array(baseline_scores[metric])
        g = np.array(gasvm_scores[metric])
        # Ensure same length (take min)
        n = min(len(b), len(g))
        b, g = b[:n], g[:n]

        try:
            w_stat, w_p = wilcoxon(g, b, alternative="greater")
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        try:
            t_stat, t_p = ttest_rel(g, b)
        except Exception:
            t_stat, t_p = float("nan"), float("nan")

        results[metric] = {
            "baseline_mean": round(float(np.mean(b)), 4),
            "baseline_std":  round(float(np.std(b, ddof=1)), 4),
            "model_mean":    round(float(np.mean(g)), 4),
            "model_std":     round(float(np.std(g, ddof=1)), 4),
            "delta":         round(float(np.mean(g) - np.mean(b)), 4),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_p":    round(float(w_p), 4),
            "ttest_p":       round(float(t_p), 4),
            "significant":   bool(w_p < 0.05),
        }
        log.info(
            f"[{label}] {metric}: baseline={np.mean(b):.4f}+-{np.std(b,ddof=1):.4f} "
            f"model={np.mean(g):.4f}+-{np.std(g,ddof=1):.4f} "
            f"p={w_p:.4f} {'*SIGNIFICANT*' if w_p < 0.05 else ''}"
        )
    return results


# ── Fix 5: Convergence analysis ───────────────────────────────────────────────

def convergence_analysis(history, label, out_path):
    """
    Analyses GA convergence:
    - Fitness curve (best + avg)
    - Population diversity (std of fitness per generation)
    - Plateau detection (generations without improvement)
    - Feature count evolution
    """
    gens      = [h["generation"]      for h in history]
    best_f    = [h["best_fitness"]    for h in history]
    avg_f     = [h["avg_fitness"]     for h in history]
    worst_f   = [h["worst_fitness"]   for h in history]
    n_feat    = [h["best_n_features"] for h in history]
    diversity = [b - w for b, w in zip(best_f, worst_f)]  # fitness spread

    # Plateau detection: generations where best fitness didn't improve
    plateau_gens = []
    for i in range(1, len(best_f)):
        if best_f[i] <= best_f[i - 1] + 1e-6:
            plateau_gens.append(gens[i])

    plateau_pct = round(len(plateau_gens) / len(gens) * 100, 1)

    # First generation where best fitness reached 99% of final value
    final_best = best_f[-1]
    convergence_gen = next(
        (gens[i] for i, v in enumerate(best_f) if v >= 0.99 * final_best),
        gens[-1]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{label} — Convergence Analysis", fontsize=13, fontweight="bold")

    # 1. Fitness curve
    ax = axes[0, 0]
    ax.plot(gens, best_f, color="#4C72B0", lw=2, label="Best")
    ax.plot(gens, avg_f,  color="#DD8452", lw=1.5, linestyle="--", label="Avg")
    ax.fill_between(gens, worst_f, best_f, alpha=0.1, color="#4C72B0")
    ax.axvline(convergence_gen, color="red", linestyle=":", lw=1.5,
               label=f"99% convergence (gen {convergence_gen})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (F1 - penalty)")
    ax.set_title("Fitness over Generations")
    ax.legend(fontsize=8)

    # 2. Population diversity (fitness spread)
    ax = axes[0, 1]
    ax.plot(gens, diversity, color="#55A868", lw=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best - Worst Fitness")
    ax.set_title("Population Diversity (Fitness Spread)")
    ax.fill_between(gens, 0, diversity, alpha=0.2, color="#55A868")

    # 3. Feature count evolution
    ax = axes[1, 0]
    ax.plot(gens, n_feat, color="#8172B2", lw=2, marker="o", markersize=3)
    ax.axhline(np.mean(n_feat), color="red", linestyle="--", lw=1,
               label=f"Mean = {np.mean(n_feat):.1f}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Features Selected")
    ax.set_title("Best Chromosome Feature Count")
    ax.set_ylim(0, 14)
    ax.legend(fontsize=8)

    # 4. Improvement per generation (delta best fitness)
    ax = axes[1, 1]
    deltas = [0] + [best_f[i] - best_f[i-1] for i in range(1, len(best_f))]
    colors = ["#55A868" if d > 1e-6 else "#CCCCCC" for d in deltas]
    ax.bar(gens, deltas, color=colors, width=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Improvement")
    ax.set_title(f"Per-Generation Improvement ({plateau_pct}% plateau)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"[convergence] Saved -> {out_path}")

    return {
        "convergence_generation": convergence_gen,
        "plateau_percentage":     plateau_pct,
        "final_best_fitness":     round(final_best, 6),
        "initial_best_fitness":   round(best_f[0], 6),
        "total_improvement":      round(final_best - best_f[0], 6),
        "mean_features_selected": round(float(np.mean(n_feat)), 2),
        "final_features_selected": n_feat[-1],
        "diversity_initial":      round(diversity[0], 6),
        "diversity_final":        round(diversity[-1], 6),
    }


# ── Summary plot ──────────────────────────────────────────────────────────────

def summary_plot(baseline_rep, gasvm_rep, out_path):
    """Box plots comparing baseline vs GA-SVM across all seeds and folds."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Statistical Comparison: Baseline SVM vs GA-SVM\n(5 seeds x 10-fold CV = 50 evaluations each)",
                 fontsize=11, fontweight="bold")

    for ax, metric, label in zip(axes,
                                  ["accuracy", "f1", "auc"],
                                  ["Accuracy", "F1 Score", "AUC-ROC"]):
        data = [baseline_rep[metric], gasvm_rep[metric]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        ax.set_xticklabels(["Baseline SVM", "GA-SVM"])
        ax.set_title(label)
        ax.set_ylabel(label)

        # Annotate means
        for i, d in enumerate([baseline_rep[metric], gasvm_rep[metric]], 1):
            ax.text(i, np.mean(d), f"  {np.mean(d):.4f}", va="center", fontsize=8,
                    color="black", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"[plot] Summary box plot saved -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_data()
    log.info(f"[data] Loaded {len(X)} samples, {X.shape[1]} features")

    # ── Fix 4: 10-fold CV ────────────────────────────────────────────────────
    log.info("\n=== Fix 4: 10-Fold Cross-Validation ===")
    cv10_baseline = ten_fold_cv_baseline(X, y)
    cv10_gasvm    = ten_fold_cv_gasvm(X, y)

    for label, scores in [("Baseline SVM", cv10_baseline), ("GA-SVM", cv10_gasvm)]:
        log.info(f"\n[10-fold] {label}:")
        for metric in ["accuracy", "f1", "auc"]:
            mean, std, lo, hi = ci95(scores[metric])
            log.info(f"  {metric}: {mean:.4f} +/- {std:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # ── Fix 1: 5-seed repeated evaluation ───────────────────────────────────
    log.info("\n=== Fix 1: 5-Seed Repeated Evaluation ===")
    rep_baseline = repeated_eval_baseline(X, y)
    rep_gasvm    = repeated_eval_gasvm(X, y)

    # ── Fix 1: Wilcoxon significance tests ───────────────────────────────────
    log.info("\n=== Fix 1: Statistical Significance Tests ===")
    sig_gasvm = significance_tests(rep_baseline, rep_gasvm, "GA-SVM vs Baseline")

    # ── Fix 5: Convergence analysis ──────────────────────────────────────────
    log.info("\n=== Fix 5: Convergence Analysis ===")
    with open(RESULTS_DIR / "ga_results.json") as f:
        ga_data = json.load(f)

    conv_stats = convergence_analysis(
        ga_data["history"], "GA-SVM",
        RESULTS_DIR / "ga_svm_convergence_analysis.png"
    )
    log.info(f"[convergence] GA-SVM converged at generation {conv_stats['convergence_generation']}")
    log.info(f"[convergence] Plateau: {conv_stats['plateau_percentage']}% of generations")
    log.info(f"[convergence] Total fitness improvement: {conv_stats['total_improvement']:.4f}")

    # ── Summary plot ─────────────────────────────────────────────────────────
    summary_plot(rep_baseline, rep_gasvm, RESULTS_DIR / "statistical_summary.png")

    # ── Build and save full report ────────────────────────────────────────────
    report = {
        "ten_fold_cv": {
            "baseline": {
                metric: {
                    "mean":  round(float(np.mean(cv10_baseline[metric])), 4),
                    "std":   round(float(np.std(cv10_baseline[metric], ddof=1)), 4),
                    "ci95":  [round(ci95(cv10_baseline[metric])[2], 4),
                              round(ci95(cv10_baseline[metric])[3], 4)],
                    "scores": [round(s, 4) for s in cv10_baseline[metric]],
                }
                for metric in ["accuracy", "f1", "auc"]
            },
            "gasvm": {
                metric: {
                    "mean":  round(float(np.mean(cv10_gasvm[metric])), 4),
                    "std":   round(float(np.std(cv10_gasvm[metric], ddof=1)), 4),
                    "ci95":  [round(ci95(cv10_gasvm[metric])[2], 4),
                              round(ci95(cv10_gasvm[metric])[3], 4)],
                    "scores": [round(s, 4) for s in cv10_gasvm[metric]],
                }
                for metric in ["accuracy", "f1", "auc"]
            },
        },
        "repeated_eval": {
            "n_seeds":   len(SEEDS),
            "seeds":     SEEDS,
            "cv_folds":  CV_FOLDS,
            "n_evals":   len(SEEDS) * CV_FOLDS,
            "baseline": {
                metric: {
                    "mean": round(float(np.mean(rep_baseline[metric])), 4),
                    "std":  round(float(np.std(rep_baseline[metric], ddof=1)), 4),
                }
                for metric in ["accuracy", "f1", "auc"]
            },
            "gasvm": {
                metric: {
                    "mean": round(float(np.mean(rep_gasvm[metric])), 4),
                    "std":  round(float(np.std(rep_gasvm[metric], ddof=1)), 4),
                }
                for metric in ["accuracy", "f1", "auc"]
            },
        },
        "significance_tests": {
            "gasvm_vs_baseline": sig_gasvm,
        },
        "convergence": {
            "gasvm": conv_stats,
        },
    }

    with open(RESULTS_DIR / "statistical_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── Print paper-ready table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAPER-READY RESULTS TABLE (10-fold CV, mean +/- std)")
    print("=" * 70)
    print(f"{'Metric':<12} {'Baseline SVM':>20} {'GA-SVM':>20} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)
    for metric in ["accuracy", "f1", "auc"]:
        bm = np.mean(cv10_baseline[metric])
        bs = np.std(cv10_baseline[metric], ddof=1)
        gm = np.mean(cv10_gasvm[metric])
        gs = np.std(cv10_gasvm[metric], ddof=1)
        p  = sig_gasvm[metric]["wilcoxon_p"]
        sig = "*" if p < 0.05 else ""
        print(f"{metric:<12} {bm:.4f} +/- {bs:.4f}   {gm:.4f} +/- {gs:.4f}   {p:.4f}  {sig}")
    print(f"\n{'Features':<12} {'13/13':>20} {str(cv10_gasvm.get('n_features','?'))+'/13':>20}")
    print(f"{'C':<12} {'1.0':>20} {str(cv10_gasvm.get('C','?')):>20}")
    print(f"{'gamma':<12} {'scale':>20} {str(cv10_gasvm.get('gamma','?')):>20}")
    print("=" * 70)
    print(f"\n[OK] Statistical analysis complete.")
    print(f"     Report saved -> {RESULTS_DIR / 'statistical_report.json'}")
    print(f"     Plots saved  -> statistical_summary.png, ga_svm_convergence_analysis.png")
