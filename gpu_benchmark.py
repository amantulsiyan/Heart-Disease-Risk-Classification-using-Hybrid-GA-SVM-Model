"""
GPU Benchmark
Measures wall-clock time for CPU vs cuML vs CuPy fitness evaluation
across different population sizes. Produces the speedup table and
chart used in the paper.

Run AFTER prepare_data.py:
    python gpu_benchmark.py
"""

import numpy as np
import json
import time
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

POP_SIZES = [10, 20, 30, 60, 100]
N_REPEATS = 3   # average over N runs per config for stable timing


def time_evaluator(mode, pop_size, X_train, y_train, n_repeats=N_REPEATS):
    from genetic_algorithm import GeneticAlgorithm, GAConfig
    from cuda_accelerator import FitnessEvaluator

    times = []
    for _ in range(n_repeats):
        cfg = GAConfig(pop_size=pop_size, n_generations=1,
                       evaluator_mode=mode, random_seed=42)
        ga  = GeneticAlgorithm(cfg)
        ga.initialise()
        ev  = FitnessEvaluator(mode=mode, cv_folds=5, random_seed=42)
        t0  = time.perf_counter()
        ev.evaluate_population(ga.population, X_train, y_train)
        times.append(time.perf_counter() - t0)

    return round(float(np.mean(times)), 3), round(float(np.std(times)), 3)


def run_benchmark(X_train, y_train):
    from cuda_accelerator import GPU_INFO

    log.info(f"[bench] GPU info: {GPU_INFO}")

    # Determine which modes to test
    modes = ["cpu"]
    if GPU_INFO["cuml_available"]:
        modes.append("cuml")
    if GPU_INFO["cupy_available"]:
        modes.append("cupy")

    if len(modes) == 1:
        log.warning("[bench] No GPU detected — only CPU results will be recorded.")
        log.warning("[bench] Run on a machine with NVIDIA GPU + RAPIDS for full benchmark.")

    results = {mode: {"times": [], "stds": []} for mode in modes}

    for pop_size in POP_SIZES:
        log.info(f"\n[bench] Population size: {pop_size}")
        for mode in modes:
            mean_t, std_t = time_evaluator(mode, pop_size, X_train, y_train)
            results[mode]["times"].append(mean_t)
            results[mode]["stds"].append(std_t)
            log.info(f"  {mode:6s}: {mean_t:.3f}s +/- {std_t:.3f}s")

    # Compute speedup relative to CPU
    speedups = {}
    for mode in modes:
        if mode == "cpu":
            continue
        speedups[mode] = [
            round(cpu_t / gpu_t, 2) if gpu_t > 0 else 0
            for cpu_t, gpu_t in zip(results["cpu"]["times"], results[mode]["times"])
        ]

    return results, speedups, modes


def speedup_plot(results, speedups, modes, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GPU-Accelerated GA-SVM: CPU vs GPU Fitness Evaluation",
                 fontsize=13, fontweight="bold")

    colors = {"cpu": "#4C72B0", "cuml": "#DD8452", "cupy": "#55A868"}
    labels = {"cpu": "CPU (joblib)", "cuml": "cuML GPU", "cupy": "CuPy GPU"}

    # Left: wall-clock time
    ax = axes[0]
    for mode in modes:
        ax.plot(POP_SIZES, results[mode]["times"],
                marker="o", lw=2, color=colors.get(mode, "gray"),
                label=labels.get(mode, mode))
        ax.fill_between(
            POP_SIZES,
            [t - s for t, s in zip(results[mode]["times"], results[mode]["stds"])],
            [t + s for t, s in zip(results[mode]["times"], results[mode]["stds"])],
            alpha=0.15, color=colors.get(mode, "gray")
        )
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Time per Generation (s)")
    ax.set_title("Wall-Clock Time vs Population Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: speedup factor
    ax = axes[1]
    if speedups:
        for mode, sp in speedups.items():
            ax.plot(POP_SIZES, sp, marker="s", lw=2,
                    color=colors.get(mode, "gray"),
                    label=f"{labels.get(mode, mode)} speedup")
        ax.axhline(1.0, color="gray", linestyle="--", lw=1, label="CPU baseline")
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Speedup Factor (x)")
        ax.set_title("GPU Speedup over CPU")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No GPU available\nRun on GPU machine",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="gray")
        ax.set_title("GPU Speedup (requires GPU)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"[bench] Speedup chart saved -> {out_path}")


def print_table(results, speedups, modes):
    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS — Time per generation (seconds, mean over 3 runs)")
    print("=" * 65)
    header = f"{'Pop':>6}" + "".join(f"  {labels.get(m, m):>12}" for m in modes)
    if speedups:
        header += "".join(f"  {'Speedup('+m+')':>14}" for m in speedups)
    print(header)
    print("-" * 65)
    labels = {"cpu": "CPU", "cuml": "cuML GPU", "cupy": "CuPy GPU"}
    for i, ps in enumerate(POP_SIZES):
        row = f"{ps:>6}"
        for mode in modes:
            row += f"  {results[mode]['times'][i]:>12.3f}"
        for mode, sp in speedups.items():
            row += f"  {sp[i]:>14.2f}x"
        print(row)
    print("=" * 65)


if __name__ == "__main__":
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    log.info(f"[bench] Dataset: {X_train.shape}")
    log.info(f"[bench] Population sizes: {POP_SIZES}")
    log.info(f"[bench] Repeats per config: {N_REPEATS}")

    results, speedups, modes = run_benchmark(X_train, y_train)

    print_table(results, speedups, modes)

    speedup_plot(results, speedups, modes,
                 RESULTS_DIR / "gpu_speedup.png")

    output = {
        "pop_sizes":  POP_SIZES,
        "n_repeats":  N_REPEATS,
        "modes":      modes,
        "results":    results,
        "speedups":   speedups,
    }
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"[bench] Results saved -> {RESULTS_DIR / 'benchmark.json'}")
    log.info("[bench] Done.")
