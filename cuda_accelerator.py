"""
Day 6 — CUDA Acceleration Module
GPU-accelerated fitness evaluation for the GA population.

Two modes:
  1. cuML mode  — uses RAPIDS cuML SVC (drop-in GPU SVM)
  2. CuPy mode  — custom RBF kernel matrix on GPU, CPU SMO solver
  3. CPU mode   — fallback, identical to baseline GA

Usage in genetic_algorithm.py:
    from backend.models.cuda_accelerator import FitnessEvaluator
    evaluator = FitnessEvaluator(mode="cuml")   # or "cupy", "cpu"
    fitnesses = evaluator.evaluate_population(population, X_train, y_train)
"""

import numpy as np
import time
import logging
from typing import List, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ── GPU availability detection ────────────────────────────────────────────────

def detect_gpu() -> dict:
    info = {"cuda_available": False, "cuml_available": False,
            "cupy_available": False, "device_name": None}
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability  # will throw if no CUDA
        info["cupy_available"] = True
        info["cuda_available"]  = True
        info["device_name"]     = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        pass
    try:
        from cuml.svm import SVC as cuSVC
        info["cuml_available"] = True
    except Exception:
        pass
    return info


GPU_INFO = detect_gpu()
log.info(f"[CUDA] GPU status: {GPU_INFO}")


# ── RBF Kernel matrix on GPU (CuPy) ──────────────────────────────────────────

def rbf_kernel_gpu(X1, X2, gamma: float):
    """
    Computes K(X1, X2) = exp(-gamma * ||x1 - x2||^2) entirely on GPU.
    X1: (n, d), X2: (m, d) — cupy arrays
    Returns: (n, m) cupy array
    """
    import cupy as cp
    # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1·x2^T
    X1_sq  = cp.sum(X1 ** 2, axis=1, keepdims=True)   # (n,1)
    X2_sq  = cp.sum(X2 ** 2, axis=1, keepdims=True).T  # (1,m)
    cross  = cp.dot(X1, X2.T)                           # (n,m)
    dist_sq = X1_sq + X2_sq - 2.0 * cross
    dist_sq = cp.maximum(dist_sq, 0.0)                  # numerical safety
    return cp.exp(-gamma * dist_sq)


# ── Fitness evaluators ────────────────────────────────────────────────────────

class CPUEvaluator:
    """Standard CPU evaluator using sklearn + joblib parallelism."""

    def __init__(self, cv_folds=5, n_jobs=-1, random_seed=42):
        self.cv_folds    = cv_folds
        self.n_jobs      = n_jobs
        self.random_seed = random_seed

    def _single_fitness(self, ch, X_train, y_train) -> float:
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        mask = ch.feature_mask()
        if mask.sum() < 2:
            return 0.0
        X_sub = X_train[:, mask]
        clf = SVC(kernel="rbf", C=ch.C(), gamma=ch.gamma(),
                  probability=False, random_state=self.random_seed)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=self.random_seed)
        try:
            scores = cross_val_score(clf, X_sub, y_train, cv=skf,
                                      scoring="f1", n_jobs=1)
            return float(scores.mean())
        except Exception:
            return 0.0

    def evaluate_population(self, population, X_train, y_train) -> List[float]:
        from joblib import Parallel, delayed
        t0 = time.perf_counter()
        fitnesses = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._single_fitness)(ch, X_train, y_train)
            for ch in population
        )
        log.debug(f"[CPU eval] {len(population)} chromosomes in "
                  f"{time.perf_counter()-t0:.2f}s")
        return fitnesses


class CuMLEvaluator:
    """
    GPU evaluator using RAPIDS cuML SVC.
    Moves data to GPU once; each evaluation slices the GPU array directly.
    """

    def __init__(self, cv_folds=5, random_seed=42):
        self.cv_folds    = cv_folds
        self.random_seed = random_seed
        self._X_gpu      = None
        self._y_gpu      = None

    def _upload_data(self, X_train, y_train):
        import cupy as cp
        if self._X_gpu is None:
            self._X_gpu = cp.asarray(X_train, dtype=cp.float32)
            self._y_gpu = cp.asarray(y_train, dtype=cp.int32)
            log.info(f"[cuML] Data uploaded to GPU: {self._X_gpu.shape}")

    def _single_fitness(self, ch, fold_indices) -> float:
        import cupy as cp
        from cuml.svm import SVC as cuSVC

        mask = ch.feature_mask()
        if mask.sum() < 2:
            return 0.0

        X_sub = self._X_gpu[:, mask]
        scores = []
        for train_idx, val_idx in fold_indices:
            X_tr = X_sub[train_idx]
            y_tr = self._y_gpu[train_idx]
            X_vl = X_sub[val_idx]
            y_vl_np = self._y_gpu[val_idx].get()

            clf = cuSVC(kernel="rbf", C=float(ch.C()), gamma=float(ch.gamma()))
            try:
                clf.fit(X_tr, y_tr)
                y_pred_np = clf.predict(X_vl).get().astype(int)
                from sklearn.metrics import f1_score
                scores.append(f1_score(y_vl_np, y_pred_np, zero_division=0))
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores))

    def evaluate_population(self, population, X_train, y_train) -> List[float]:
        from sklearn.model_selection import StratifiedKFold
        self._upload_data(X_train, y_train)

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=42)
        fold_indices = list(skf.split(X_train, y_train))

        t0 = time.perf_counter()
        fitnesses = [self._single_fitness(ch, fold_indices) for ch in population]
        log.info(f"[cuML eval] {len(population)} chromosomes in "
                 f"{time.perf_counter()-t0:.2f}s")
        return fitnesses


class CuPyKernelEvaluator:
    """
    GPU evaluator: computes the full RBF kernel matrix on GPU,
    then runs sklearn SVC in precomputed-kernel mode on CPU.
    Useful when cuML is not installed but CuPy is.
    """

    def __init__(self, cv_folds=5, random_seed=42):
        self.cv_folds    = cv_folds
        self.random_seed = random_seed

    def _single_fitness(self, ch, X_train, y_train) -> float:
        import cupy as cp
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold

        mask = ch.feature_mask()
        if mask.sum() < 2:
            return 0.0

        X_sub_gpu = cp.asarray(X_train[:, mask], dtype=cp.float32)
        gamma     = ch.gamma()
        K_gpu     = rbf_kernel_gpu(X_sub_gpu, X_sub_gpu, gamma)
        K_cpu     = cp.asnumpy(K_gpu).astype(np.float64)

        clf = SVC(kernel="precomputed", C=ch.C(), probability=False,
                  random_state=self.random_seed)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=self.random_seed)
        scores = []
        try:
            for tr, vl in skf.split(K_cpu, y_train):
                clf.fit(K_cpu[np.ix_(tr, tr)], y_train[tr])
                y_pred = clf.predict(K_cpu[np.ix_(vl, tr)])
                from sklearn.metrics import f1_score
                scores.append(f1_score(y_train[vl], y_pred, zero_division=0))
        except Exception:
            return 0.0
        return float(np.mean(scores))

    def evaluate_population(self, population, X_train, y_train) -> List[float]:
        t0 = time.perf_counter()
        fitnesses = [self._single_fitness(ch, X_train, y_train) for ch in population]
        log.info(f"[CuPy eval] {len(population)} chromosomes in "
                 f"{time.perf_counter()-t0:.2f}s")
        return fitnesses


# ── Public factory ─────────────────────────────────────────────────────────────

class FitnessEvaluator:
    """
    Auto-selects the best available evaluator.
    mode: "auto" | "cuml" | "cupy" | "cpu"
    """

    def __new__(cls, mode="auto", **kwargs):
        if mode == "auto":
            if GPU_INFO["cuml_available"]:
                log.info("[FitnessEvaluator] Using cuML (GPU)")
                return CuMLEvaluator(**kwargs)
            elif GPU_INFO["cupy_available"]:
                log.info("[FitnessEvaluator] Using CuPy kernel (GPU)")
                return CuPyKernelEvaluator(**kwargs)
            else:
                log.info("[FitnessEvaluator] No GPU found, using CPU")
                return CPUEvaluator(**kwargs)
        elif mode == "cuml":
            return CuMLEvaluator(**kwargs)
        elif mode == "cupy":
            return CuPyKernelEvaluator(**kwargs)
        else:
            return CPUEvaluator(**kwargs)


# ── Benchmark utility ──────────────────────────────────────────────────────────

def benchmark_evaluators(X_train, y_train, pop_sizes=(10, 30, 60)):
    """
    Times CPU vs GPU evaluator across different population sizes.
    Run this after training is done to generate the speedup chart.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, str(Path(__file__).parent))
    from genetic_algorithm import GeneticAlgorithm, GAConfig

    results = {}
    for mode in (["cpu"] + (["cuml"] if GPU_INFO["cuml_available"] else [])):
        mode_times = []
        for ps in pop_sizes:
            cfg = GAConfig(pop_size=ps, n_generations=1, n_jobs=-1)
            ga  = GeneticAlgorithm(cfg)
            ga.initialise()
            ev  = FitnessEvaluator(mode=mode)
            t0  = time.perf_counter()
            ev.evaluate_population(ga.population, X_train, y_train)
            elapsed = time.perf_counter() - t0
            mode_times.append(round(elapsed, 3))
            log.info(f"[bench] mode={mode} pop={ps} time={elapsed:.3f}s")
        results[mode] = mode_times

    return {"pop_sizes": list(pop_sizes), "times": results}


if __name__ == "__main__":
    import json
    from pathlib import Path
    DATA_DIR    = Path(__file__).parent / "data"
    RESULTS_DIR = Path(__file__).parent / "results"

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    print(f"\n[CUDA] GPU info: {GPU_INFO}")

    bench = benchmark_evaluators(X_train, y_train, pop_sizes=[10, 30, 60, 100])
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(bench, f, indent=2)
    print(f"\n[Day 6] ✓  Benchmark saved. Results: {bench}")
