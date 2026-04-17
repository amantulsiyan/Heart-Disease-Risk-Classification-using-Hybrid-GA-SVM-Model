"""
Day 3–4 — Genetic Algorithm Engine
Full GA implementation for simultaneous feature selection
and SVM hyperparameter optimisation.

Chromosome layout (15 genes):
  [0..12]  — binary feature mask (1 = use this feature)
  [13]     — C index into C_VALUES grid
  [14]     — gamma index into GAMMA_VALUES grid

Run:  python backend/models/genetic_algorithm.py
"""

import numpy as np
import json
import time
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

# ── Hyperparameter search grids ─────────────────────────────────────────────
C_VALUES     = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
GAMMA_VALUES = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class GAConfig:
    pop_size:        int   = 60
    n_generations:   int   = 80
    crossover_rate:  float = 0.80
    mutation_rate:   float = 0.02
    tournament_k:    int   = 3
    elitism_n:       int   = 2        # best N chromosomes survive unchanged
    cv_folds:        int   = 5
    feature_penalty: float = 0.001    # penalise using too many features
    min_features:    int   = 2        # enforce at least N features selected
    random_seed:     int   = 42
    n_jobs:          int   = -1       # -1 = all CPU cores


@dataclass
class Chromosome:
    genes: np.ndarray                 # length 15: 13 feature bits + 2 param indices
    fitness: float = -1.0
    n_features: int = 0

    def feature_mask(self) -> np.ndarray:
        return self.genes[:13].astype(bool)

    def C(self) -> float:
        idx = int(self.genes[13]) % len(C_VALUES)
        return C_VALUES[idx]

    def gamma(self) -> float:
        idx = int(self.genes[14]) % len(GAMMA_VALUES)
        return GAMMA_VALUES[idx]

    def n_selected(self) -> int:
        return int(self.genes[:13].sum())

    def to_dict(self) -> dict:
        return {
            "feature_mask": self.genes[:13].astype(int).tolist(),
            "C_index":      int(self.genes[13]),
            "gamma_index":  int(self.genes[14]),
            "C":            self.C(),
            "gamma":        self.gamma(),
            "fitness":      round(self.fitness, 6),
            "n_features":   self.n_selected(),
        }


@dataclass
class GenerationStats:
    generation:   int
    best_fitness: float
    avg_fitness:  float
    worst_fitness: float
    best_n_features: int
    elapsed_s:    float


# ── Core GA ──────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    def __init__(self, cfg: GAConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.population: List[Chromosome] = []
        self.history: List[GenerationStats] = []
        self.best_ever: Optional[Chromosome] = None

    # ── Initialisation ───────────────────────────────────────────────────────

    def _random_chromosome(self) -> Chromosome:
        """
        Random chromosome with guaranteed minimum features selected.
        """
        genes = np.zeros(15, dtype=float)
        # Feature bits: random binary, but enforce min_features
        n_on = self.rng.integers(self.cfg.min_features, 14)
        on_idx = self.rng.choice(13, size=int(n_on), replace=False)
        genes[on_idx] = 1.0
        # Hyperparameter indices
        genes[13] = self.rng.integers(0, len(C_VALUES))
        genes[14] = self.rng.integers(0, len(GAMMA_VALUES))
        return Chromosome(genes=genes)

    def initialise(self):
        self.population = [
            self._random_chromosome() for _ in range(self.cfg.pop_size)
        ]
        log.info(f"[GA] Initialised population of {self.cfg.pop_size} chromosomes.")

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _compute_fitness(self, ch: Chromosome, X_train, y_train) -> float:
        """
        Fitness = mean 5-fold CV F1 score − feature-count penalty.
        Returns 0 if no features selected.
        """
        mask = ch.feature_mask()
        if mask.sum() < self.cfg.min_features:
            return 0.0

        X_sub = X_train[:, mask]
        clf = SVC(
            kernel="rbf",
            C=ch.C(),
            gamma=ch.gamma(),
            probability=False,
            random_state=self.cfg.random_seed,
            cache_size=300,
        )
        try:
            skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True,
                                   random_state=self.cfg.random_seed)
            scores = cross_val_score(clf, X_sub, y_train, cv=skf, scoring="f1", n_jobs=1)
            f1 = scores.mean()
        except Exception:
            return 0.0

        # Sparsity penalty: discourage using all 13 features
        penalty = self.cfg.feature_penalty * mask.sum()
        return float(max(0.0, f1 - penalty))

    def evaluate_population(self, X_train, y_train):
        """
        CPU-parallel evaluation using joblib.
        Replace the inner SVC with cuML SVC for GPU acceleration.
        """
        from joblib import Parallel, delayed

        fitnesses = Parallel(n_jobs=self.cfg.n_jobs, prefer="threads")(
            delayed(self._compute_fitness)(ch, X_train, y_train)
            for ch in self.population
        )
        for ch, fit in zip(self.population, fitnesses):
            ch.fitness = fit
            ch.n_features = ch.n_selected()

    # ── Selection ─────────────────────────────────────────────────────────────

    def tournament_select(self) -> Chromosome:
        candidates = self.rng.choice(len(self.population), size=self.cfg.tournament_k,
                                      replace=False)
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[best]

    # ── Crossover ─────────────────────────────────────────────────────────────

    def crossover(self, parent_a: Chromosome, parent_b: Chromosome
                  ) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() > self.cfg.crossover_rate:
            return (Chromosome(genes=parent_a.genes.copy()),
                    Chromosome(genes=parent_b.genes.copy()))

        # Two-point crossover
        pts = sorted(self.rng.choice(15, size=2, replace=False))
        p, q = pts[0], pts[1]
        g_a = np.concatenate([parent_a.genes[:p], parent_b.genes[p:q], parent_a.genes[q:]])
        g_b = np.concatenate([parent_b.genes[:p], parent_a.genes[p:q], parent_b.genes[q:]])
        return Chromosome(genes=g_a.copy()), Chromosome(genes=g_b.copy())

    # ── Mutation ──────────────────────────────────────────────────────────────

    def mutate(self, ch: Chromosome) -> Chromosome:
        genes = ch.genes.copy()
        for i in range(13):   # feature bits: bit-flip
            if self.rng.random() < self.cfg.mutation_rate:
                genes[i] = 1.0 - genes[i]
        # Hyperparameter genes: random reset with small probability
        if self.rng.random() < self.cfg.mutation_rate * 2:
            genes[13] = self.rng.integers(0, len(C_VALUES))
        if self.rng.random() < self.cfg.mutation_rate * 2:
            genes[14] = self.rng.integers(0, len(GAMMA_VALUES))

        # Enforce minimum features
        if genes[:13].sum() < self.cfg.min_features:
            forced = self.rng.choice(13, size=self.cfg.min_features, replace=False)
            genes[forced] = 1.0
        return Chromosome(genes=genes)

    # ── Next generation ───────────────────────────────────────────────────────

    def evolve(self):
        new_pop: List[Chromosome] = []

        # Elitism: carry over the best N chromosomes unchanged
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        for elite in sorted_pop[:self.cfg.elitism_n]:
            new_pop.append(Chromosome(genes=elite.genes.copy(), fitness=elite.fitness,
                                       n_features=elite.n_features))

        while len(new_pop) < self.cfg.pop_size:
            pa = self.tournament_select()
            pb = self.tournament_select()
            ca, cb = self.crossover(pa, pb)
            ca = self.mutate(ca)
            cb = self.mutate(cb)
            new_pop.append(ca)
            if len(new_pop) < self.cfg.pop_size:
                new_pop.append(cb)

        self.population = new_pop[:self.cfg.pop_size]

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, X_train, y_train, progress_callback=None) -> Chromosome:
        self.initialise()
        t_start = time.perf_counter()

        for gen in range(self.cfg.n_generations):
            t_gen = time.perf_counter()
            self.evaluate_population(X_train, y_train)

            fitnesses = [c.fitness for c in self.population]
            best_ch   = max(self.population, key=lambda c: c.fitness)

            if self.best_ever is None or best_ch.fitness > self.best_ever.fitness:
                self.best_ever = Chromosome(genes=best_ch.genes.copy(),
                                             fitness=best_ch.fitness,
                                             n_features=best_ch.n_features)

            stats = GenerationStats(
                generation    = gen + 1,
                best_fitness  = float(best_ch.fitness),
                avg_fitness   = float(np.mean(fitnesses)),
                worst_fitness = float(min(fitnesses)),
                best_n_features = best_ch.n_selected(),
                elapsed_s     = round(time.perf_counter() - t_gen, 3),
            )
            self.history.append(stats)

            log.info(
                f"Gen {gen+1:3d}/{self.cfg.n_generations} | "
                f"Best: {stats.best_fitness:.4f} | "
                f"Avg: {stats.avg_fitness:.4f} | "
                f"Features: {stats.best_n_features}/13 | "
                f"Time: {stats.elapsed_s}s"
            )

            if progress_callback:
                progress_callback(gen + 1, stats, self.best_ever)

            if gen < self.cfg.n_generations - 1:
                self.evolve()

        total = round(time.perf_counter() - t_start, 2)
        log.info(f"\n[GA] Finished in {total}s | Best fitness: {self.best_ever.fitness:.4f}")
        return self.best_ever

    # ── Serialisation ─────────────────────────────────────────────────────────

    def history_to_list(self):
        return [
            {
                "generation":    s.generation,
                "best_fitness":  round(s.best_fitness, 6),
                "avg_fitness":   round(s.avg_fitness, 6),
                "worst_fitness": round(s.worst_fitness, 6),
                "best_n_features": s.best_n_features,
                "elapsed_s":     s.elapsed_s,
            }
            for s in self.history
        ]


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load data
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    cfg = GAConfig(pop_size=60, n_generations=80)
    ga  = GeneticAlgorithm(cfg)
    best = ga.run(X_train, y_train)

    print("\n── Best chromosome ─────────────────")
    print(json.dumps(best.to_dict(), indent=2))

    # Save
    history = ga.history_to_list()
    result = {
        "best_chromosome": best.to_dict(),
        "history":         history,
        "config":          cfg.__dict__,
        "C_values":        C_VALUES,
        "gamma_values":    GAMMA_VALUES,
    }
    with open(RESULTS_DIR / "ga_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / "best_chromosome.pkl", "wb") as f:
        pickle.dump(best, f)

    # Fitness curve plot
    gens = [s["generation"] for s in history]
    best_f = [s["best_fitness"] for s in history]
    avg_f  = [s["avg_fitness"] for s in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, best_f, label="Best fitness", color="steelblue", lw=2)
    ax.plot(gens, avg_f,  label="Avg fitness",  color="orange",    lw=1.5, linestyle="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (F1 − penalty)")
    ax.set_title("GA Convergence Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ga_convergence.png", dpi=150)
    plt.close()

    print(f"\n[Day 3–4] ✓  GA complete. Best: F1={best.fitness:.4f}, "
          f"features={best.n_selected()}/13, C={best.C()}, γ={best.gamma()}")
