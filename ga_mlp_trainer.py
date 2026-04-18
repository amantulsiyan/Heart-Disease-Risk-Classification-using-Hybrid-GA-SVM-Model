"""
GA-MLP Trainer
Genetic Algorithm optimising simultaneous feature selection AND
neural-network hyperparameters for an MLP classifier.

Chromosome layout (17 genes):
  [0..12]  — binary feature mask (1 = use this feature)
  [13]     — learning rate index
  [14]     — hidden layer size index
  [15]     — dropout index
  [16]     — network depth index

Each fitness call trains a PyTorch MLP for 200 epochs on GPU (5-fold CV).
Run:  python ga_mlp_trainer.py
"""

import numpy as np
import json
import pickle
import time
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"[device] Using: {DEVICE}")
if DEVICE.type == "cuda":
    log.info(f"[device] GPU: {torch.cuda.get_device_name(0)}")

# ── Hyperparameter search grids ──────────────────────────────────────────────
LR_VALUES      = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
HIDDEN_VALUES  = [32, 64, 128, 256, 512]
DROPOUT_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]
DEPTH_VALUES   = [1, 2, 3, 4]          # number of hidden layers

EPOCHS         = 200
BATCH_SIZE     = 32
FEATURE_PENALTY = 0.001
MIN_FEATURES    = 2


# ── MLP model ────────────────────────────────────────────────────────────────

def build_mlp(n_features: int, hidden_size: int, depth: int, dropout: float) -> nn.Module:
    layers = []
    in_size = n_features
    for _ in range(depth):
        layers += [nn.Linear(in_size, hidden_size), nn.BatchNorm1d(hidden_size),
                   nn.ReLU(), nn.Dropout(dropout)]
        in_size = hidden_size
    layers.append(nn.Linear(in_size, 1))   # binary output (logit)
    return nn.Sequential(*layers)


# ── Single fold training ──────────────────────────────────────────────────────

def train_fold(X_tr: torch.Tensor, y_tr: torch.Tensor,
               X_vl: torch.Tensor, y_vl: torch.Tensor,
               hidden: int, depth: int, dropout: float, lr: float) -> float:
    model = build_mlp(X_tr.shape[1], hidden, depth, dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_vl).squeeze(1)
        preds  = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
        labels = y_vl.cpu().numpy().astype(int)

    return float(f1_score(labels, preds, zero_division=0))


# ── Fitness function ──────────────────────────────────────────────────────────

def compute_fitness(genes: np.ndarray, X_train: np.ndarray,
                    y_train: np.ndarray, cv_folds: int = 5) -> float:
    mask    = genes[:13].astype(bool)
    n_sel   = int(mask.sum())
    if n_sel < MIN_FEATURES:
        return 0.0

    lr      = LR_VALUES[int(genes[13]) % len(LR_VALUES)]
    hidden  = HIDDEN_VALUES[int(genes[14]) % len(HIDDEN_VALUES)]
    dropout = DROPOUT_VALUES[int(genes[15]) % len(DROPOUT_VALUES)]
    depth   = DEPTH_VALUES[int(genes[16]) % len(DEPTH_VALUES)]

    X_sub = X_train[:, mask].astype(np.float32)
    skf   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []

    for tr_idx, vl_idx in skf.split(X_sub, y_train):
        X_tr = torch.tensor(X_sub[tr_idx]).to(DEVICE)
        y_tr = torch.tensor(y_train[tr_idx].astype(np.float32)).to(DEVICE)
        X_vl = torch.tensor(X_sub[vl_idx]).to(DEVICE)
        y_vl = torch.tensor(y_train[vl_idx].astype(np.float32)).to(DEVICE)
        try:
            scores.append(train_fold(X_tr, y_tr, X_vl, y_vl, hidden, depth, dropout, lr))
        except Exception:
            scores.append(0.0)

    return float(max(0.0, np.mean(scores) - FEATURE_PENALTY * n_sel))


# ── GA dataclasses ────────────────────────────────────────────────────────────

@dataclass
class MLPChromosome:
    genes:      np.ndarray
    fitness:    float = -1.0

    def feature_mask(self):  return self.genes[:13].astype(bool)
    def lr(self):            return LR_VALUES[int(self.genes[13]) % len(LR_VALUES)]
    def hidden(self):        return HIDDEN_VALUES[int(self.genes[14]) % len(HIDDEN_VALUES)]
    def dropout(self):       return DROPOUT_VALUES[int(self.genes[15]) % len(DROPOUT_VALUES)]
    def depth(self):         return DEPTH_VALUES[int(self.genes[16]) % len(DEPTH_VALUES)]
    def n_selected(self):    return int(self.genes[:13].sum())

    def to_dict(self):
        return {
            "feature_mask": self.genes[:13].astype(int).tolist(),
            "lr":           self.lr(),
            "hidden_size":  self.hidden(),
            "dropout":      self.dropout(),
            "depth":        self.depth(),
            "fitness":      round(self.fitness, 6),
            "n_features":   self.n_selected(),
        }


@dataclass
class GAMLPConfig:
    pop_size:       int   = 30
    n_generations:  int   = 40
    crossover_rate: float = 0.80
    mutation_rate:  float = 0.02
    tournament_k:   int   = 3
    elitism_n:      int   = 2
    cv_folds:       int   = 5
    random_seed:    int   = 42


# ── GA engine ─────────────────────────────────────────────────────────────────

class GAForMLP:
    def __init__(self, cfg: GAMLPConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.population: List[MLPChromosome] = []
        self.history = []
        self.best_ever: Optional[MLPChromosome] = None

    def _random_chromosome(self) -> MLPChromosome:
        genes = np.zeros(17, dtype=float)
        n_on  = self.rng.integers(MIN_FEATURES, 14)
        genes[self.rng.choice(13, size=int(n_on), replace=False)] = 1.0
        genes[13] = self.rng.integers(0, len(LR_VALUES))
        genes[14] = self.rng.integers(0, len(HIDDEN_VALUES))
        genes[15] = self.rng.integers(0, len(DROPOUT_VALUES))
        genes[16] = self.rng.integers(0, len(DEPTH_VALUES))
        return MLPChromosome(genes=genes)

    def _tournament(self) -> MLPChromosome:
        idxs = self.rng.choice(len(self.population), size=self.cfg.tournament_k, replace=False)
        return self.population[max(idxs, key=lambda i: self.population[i].fitness)]

    def _crossover(self, a: MLPChromosome, b: MLPChromosome) -> Tuple[MLPChromosome, MLPChromosome]:
        if self.rng.random() > self.cfg.crossover_rate:
            return MLPChromosome(genes=a.genes.copy()), MLPChromosome(genes=b.genes.copy())
        pts = sorted(self.rng.choice(17, size=2, replace=False))
        p, q = pts[0], pts[1]
        ga = np.concatenate([a.genes[:p], b.genes[p:q], a.genes[q:]])
        gb = np.concatenate([b.genes[:p], a.genes[p:q], b.genes[q:]])
        return MLPChromosome(genes=ga.copy()), MLPChromosome(genes=gb.copy())

    def _mutate(self, ch: MLPChromosome) -> MLPChromosome:
        genes = ch.genes.copy()
        for i in range(13):
            if self.rng.random() < self.cfg.mutation_rate:
                genes[i] = 1.0 - genes[i]
        for i, grid_len in zip([13, 14, 15, 16],
                                [len(LR_VALUES), len(HIDDEN_VALUES),
                                 len(DROPOUT_VALUES), len(DEPTH_VALUES)]):
            if self.rng.random() < self.cfg.mutation_rate * 2:
                genes[i] = self.rng.integers(0, grid_len)
        if genes[:13].sum() < MIN_FEATURES:
            genes[self.rng.choice(13, size=MIN_FEATURES, replace=False)] = 1.0
        return MLPChromosome(genes=genes)

    def _evolve(self):
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        new_pop = [MLPChromosome(genes=e.genes.copy(), fitness=e.fitness)
                   for e in sorted_pop[:self.cfg.elitism_n]]
        while len(new_pop) < self.cfg.pop_size:
            ca, cb = self._crossover(self._tournament(), self._tournament())
            new_pop.append(self._mutate(ca))
            if len(new_pop) < self.cfg.pop_size:
                new_pop.append(self._mutate(cb))
        self.population = new_pop[:self.cfg.pop_size]

    def run(self, X_train, y_train, progress_callback=None) -> MLPChromosome:
        self.population = [self._random_chromosome() for _ in range(self.cfg.pop_size)]
        log.info(f"[GA-MLP] Population: {self.cfg.pop_size} | Generations: {self.cfg.n_generations} | Device: {DEVICE}")
        t_start = time.perf_counter()

        for gen in range(self.cfg.n_generations):
            t_gen = time.perf_counter()

            for ch in self.population:
                if ch.fitness < 0:
                    ch.fitness = compute_fitness(ch.genes, X_train, y_train, self.cfg.cv_folds)

            fitnesses = [c.fitness for c in self.population]
            best_ch   = max(self.population, key=lambda c: c.fitness)

            if self.best_ever is None or best_ch.fitness > self.best_ever.fitness:
                self.best_ever = MLPChromosome(genes=best_ch.genes.copy(), fitness=best_ch.fitness)

            elapsed = round(time.perf_counter() - t_gen, 2)
            stats = {
                "generation":      gen + 1,
                "best_fitness":    round(float(best_ch.fitness), 6),
                "avg_fitness":     round(float(np.mean(fitnesses)), 6),
                "worst_fitness":   round(float(min(fitnesses)), 6),
                "best_n_features": best_ch.n_selected(),
                "elapsed_s":       elapsed,
            }
            self.history.append(stats)

            log.info(
                f"Gen {gen+1:3d}/{self.cfg.n_generations} | "
                f"Best: {stats['best_fitness']:.4f} | "
                f"Avg: {stats['avg_fitness']:.4f} | "
                f"Features: {stats['best_n_features']}/13 | "
                f"Time: {elapsed}s"
            )

            if progress_callback:
                progress_callback(gen + 1, stats, self.best_ever)

            if gen < self.cfg.n_generations - 1:
                self._evolve()
                # only unevaluated chromosomes get fitness=-1 after evolve
                for ch in self.population:
                    if ch.fitness < 0:
                        ch.fitness = -1.0

        total = round(time.perf_counter() - t_start, 2)
        log.info(f"[GA-MLP] Done in {total}s | Best fitness: {self.best_ever.fitness:.4f}")
        return self.best_ever


# ── Final model training ──────────────────────────────────────────────────────

def train_final_model(best: MLPChromosome, X_train, y_train, X_test, y_test):
    mask    = best.feature_mask()
    X_tr    = torch.tensor(X_train[:, mask].astype(np.float32)).to(DEVICE)
    y_tr    = torch.tensor(y_train.astype(np.float32)).to(DEVICE)
    X_te    = torch.tensor(X_test[:, mask].astype(np.float32)).to(DEVICE)

    model     = build_mlp(int(mask.sum()), best.hidden(), best.depth(), best.dropout()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=best.lr(), weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ds        = TensorDataset(X_tr, y_tr)
    loader    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.perf_counter()
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb).squeeze(1), yb).backward()
            optimizer.step()
        scheduler.step()
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        logits = model(X_te).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm  = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)

    print(f"\n-- GA-MLP Final Evaluation --")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Features : {int(mask.sum())}/13")
    print(f"  Depth    : {best.depth()} layers x {best.hidden()} units")
    print(f"  LR       : {best.lr()} | Dropout: {best.dropout()}")

    return model, mask, {
        "accuracy":       round(float(acc), 4),
        "f1":             round(float(f1), 4),
        "auc":            round(float(auc), 4),
        "cm":             cm.tolist(),
        "y_prob":         probs.tolist(),
        "y_pred":         preds.tolist(),
        "roc":            {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "train_time_s":   round(train_time, 4),
        "n_features_used": int(mask.sum()),
        "feature_mask":   mask.astype(int).tolist(),
        "lr":             best.lr(),
        "hidden_size":    best.hidden(),
        "dropout":        best.dropout(),
        "depth":          best.depth(),
        "ga_fitness":     best.fitness,
        "device":         str(DEVICE),
    }


def ten_fold_cv_mlp(best, X_train, y_train, X_test, y_test):
    """Fix 4: 10-fold stratified CV on full dataset for paper-quality numbers."""
    mask   = best.feature_mask()
    X_full = np.concatenate([X_train, X_test])[:, mask].astype(np.float32)
    y_full = np.concatenate([y_train, y_test])
    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc_s, f1_s, auc_s = [], [], []

    for tr, vl in skf.split(X_full, y_full):
        X_tr = torch.tensor(X_full[tr]).to(DEVICE)
        y_tr = torch.tensor(y_full[tr].astype(np.float32)).to(DEVICE)
        X_vl = torch.tensor(X_full[vl]).to(DEVICE)
        try:
            score = train_fold(X_tr, y_tr, X_vl,
                               torch.tensor(y_full[vl].astype(np.float32)).to(DEVICE),
                               best.hidden(), best.depth(), best.dropout(), best.lr())
            f1_s.append(score)
            # get probs for acc and auc
            model = build_mlp(X_tr.shape[1], best.hidden(), best.depth(), best.dropout()).to(DEVICE)
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(X_vl).squeeze(1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            acc_s.append(accuracy_score(y_full[vl], preds))
            auc_s.append(roc_auc_score(y_full[vl], probs))
        except Exception:
            acc_s.append(0.0); f1_s.append(0.0); auc_s.append(0.0)

    return {
        "cv10_accuracy_mean": round(float(np.mean(acc_s)), 4),
        "cv10_accuracy_std":  round(float(np.std(acc_s, ddof=1)), 4),
        "cv10_f1_mean":       round(float(np.mean(f1_s)), 4),
        "cv10_f1_std":        round(float(np.std(f1_s, ddof=1)), 4),
        "cv10_auc_mean":      round(float(np.mean(auc_s)), 4),
        "cv10_auc_std":       round(float(np.std(auc_s, ddof=1)), 4),
    }


# ── Comparison plot ───────────────────────────────────────────────────────────

def comparison_plot(baseline, ga_svm, ga_mlp, feature_names, mask, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("GA-MLP vs GA-SVM vs Baseline SVM", fontsize=14, fontweight="bold")

    # 1. Metric bar chart (3 models)
    ax = axes[0, 0]
    metrics = ["Accuracy", "F1 Score", "AUC-ROC"]
    x = np.arange(len(metrics))
    w = 0.25
    for i, (label, res, color) in enumerate([
        ("Baseline SVM", baseline, "#4C72B0"),
        ("GA-SVM",       ga_svm,   "#DD8452"),
        ("GA-MLP",       ga_mlp,   "#55A868"),
    ]):
        vals = [res["accuracy"], res["f1"], res["auc"]]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Metric Comparison")
    ax.legend(fontsize=8)

    # 2. GA-MLP confusion matrix
    ax = axes[0, 1]
    sns.heatmap(np.array(ga_mlp["cm"]), annot=True, fmt="d", cmap="Greens",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title("GA-MLP Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    # 3. ROC curves (all 3)
    ax = axes[1, 0]
    for label, res, color in [
        ("Baseline SVM", baseline, "#4C72B0"),
        ("GA-SVM",       ga_svm,   "#DD8452"),
        ("GA-MLP",       ga_mlp,   "#55A868"),
    ]:
        ax.plot(res["roc"]["fpr"], res["roc"]["tpr"], lw=2,
                label=f"{label} (AUC={res['auc']:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)

    # 4. GA-MLP selected features
    ax = axes[1, 1]
    colors = ["#55A868" if m else "#CCCCCC" for m in mask]
    ax.barh(feature_names, mask.astype(int), color=colors)
    ax.set_xlim(0, 1.3)
    ax.set_title(f"GA-MLP Selected Features ({int(mask.sum())}/13)")
    ax.set_xlabel("Selected (1 = yes)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved -> {out_path}")


# ── Convergence plot ──────────────────────────────────────────────────────────

def convergence_plot(history, out_path):
    gens   = [h["generation"]   for h in history]
    best_f = [h["best_fitness"] for h in history]
    avg_f  = [h["avg_fitness"]  for h in history]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gens, best_f, label="Best fitness", color="#55A868", lw=2)
    ax.plot(gens, avg_f,  label="Avg fitness",  color="orange",  lw=1.5, linestyle="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (F1 - penalty)")
    ax.set_title("GA-MLP Convergence Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Convergence saved -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(RESULTS_DIR / "baseline_results.json") as f:
        baseline = json.load(f)
    with open(RESULTS_DIR / "ga_svm_results.json") as f:
        ga_svm = json.load(f)

    feature_names = meta["feature_names"]

    cfg  = GAMLPConfig(pop_size=30, n_generations=40)
    ga   = GAForMLP(cfg)
    best = ga.run(X_train, y_train)

    print("\n-- Best chromosome --")
    print(json.dumps(best.to_dict(), indent=2))

    # Train final model on full training set
    model, mask, results = train_final_model(best, X_train, y_train, X_test, y_test)

    # Fix 4: 10-fold CV for paper-quality numbers
    log.info("[10-fold CV] Running 10-fold CV on GA-MLP best chromosome...")
    cv10 = ten_fold_cv_mlp(best, X_train, y_train, X_test, y_test)
    results.update(cv10)
    print(f"\n[10-fold CV] Accuracy: {cv10['cv10_accuracy_mean']:.4f} +/- {cv10['cv10_accuracy_std']:.4f}")
    print(f"[10-fold CV] F1:       {cv10['cv10_f1_mean']:.4f} +/- {cv10['cv10_f1_std']:.4f}")
    print(f"[10-fold CV] AUC:      {cv10['cv10_auc_mean']:.4f} +/- {cv10['cv10_auc_std']:.4f}")

    # Plots
    comparison_plot(baseline, ga_svm, results, feature_names, mask,
                    RESULTS_DIR / "ga_mlp_comparison.png")
    convergence_plot(ga.history, RESULTS_DIR / "ga_mlp_convergence.png")

    # Save
    torch.save({"model_state": model.state_dict(),
                "mask": mask,
                "hidden": best.hidden(),
                "depth":  best.depth(),
                "dropout": best.dropout()},
               RESULTS_DIR / "ga_mlp_model.pt")

    ga_mlp_output = {
        "best_chromosome": best.to_dict(),
        "history":         ga.history,
        "results":         results,
        "config": {
            "pop_size":      cfg.pop_size,
            "n_generations": cfg.n_generations,
            "epochs":        EPOCHS,
            "device":        str(DEVICE),
        },
        "lr_values":      LR_VALUES,
        "hidden_values":  HIDDEN_VALUES,
        "dropout_values": DROPOUT_VALUES,
        "depth_values":   DEPTH_VALUES,
    }
    with open(RESULTS_DIR / "ga_mlp_results.json", "w") as f:
        json.dump(ga_mlp_output, f, indent=2)

    # Head-to-head table
    print("\n-- Head-to-head (all 3 models) --")
    print(f"{'Metric':<20} {'Baseline SVM':>14} {'GA-SVM':>14} {'GA-MLP':>14}")
    print("-" * 64)
    for metric in ["accuracy", "f1", "auc"]:
        print(f"  {metric:<18} {baseline[metric]:>14.4f} {ga_svm[metric]:>14.4f} {results[metric]:>14.4f}")
    print(f"  {'features used':<18} {baseline['n_features_used']:>14} {ga_svm['n_features_used']:>14} {results['n_features_used']:>14}")
    print(f"  {'device':<18} {'CPU':>14} {'CPU':>14} {str(DEVICE):>14}")

    print(f"\n[GA-MLP] Done. Accuracy={results['accuracy']:.4f}, "
          f"F1={results['f1']:.4f}, AUC={results['auc']:.4f}, "
          f"Features={results['n_features_used']}/13")
