# CardioGA — GPU-Accelerated Genetic Algorithm for SVM Optimisation

A GPU-accelerated hybrid machine learning system that uses a **Genetic Algorithm (GA)** to simultaneously perform feature selection and SVM hyperparameter optimisation for heart disease prediction.

The core contribution is the **GPU-accelerated fitness evaluation pipeline**: by replacing CPU-based sklearn SVC with RAPIDS cuML SVC (or CuPy kernel computation), the GA evaluates each generation's population significantly faster, enabling larger population sizes and more generations within practical time budgets — leading to better feature subsets and hyperparameters.

Three models are implemented and compared:

- **Baseline SVM** — plain RBF-SVM on all 13 features, default hyperparameters, CPU
- **GA-SVM (CPU)** — GA with joblib-parallel sklearn SVC fitness evaluation
- **GA-SVM (GPU)** — same GA with cuML/CuPy GPU-accelerated fitness evaluation

Dataset: combined UCI Heart Disease (Cleveland + Hungarian + Switzerland + VA Long Beach) — **920 patients**, 13 clinical features, binary target.

---

## Core contribution

```
Standard GA-SVM:
  for each generation:
    for each chromosome:          ← serial
      sklearn SVC fit (CPU)       ← slow

GPU-Accelerated GA-SVM:
  upload X_train to GPU once      ← single transfer
  for each generation:
    for each chromosome:
      cuML SVC fit (GPU)          ← fast
      OR CuPy RBF kernel (GPU) + precomputed SVC
```

The GA itself (selection, crossover, mutation) stays on CPU — it operates on 60 chromosomes, which is trivial. Only the compute-heavy SVM fitness evaluations move to GPU.

**Paper claim:** GPU acceleration achieves Xx speedup over CPU baseline at population size 60, enabling the GA to run 80 generations in minutes instead of hours, with no change in solution quality.

---

## Chromosome layout (15 genes)

```
[ f1 f2 f3 ... f13 | C_idx | gamma_idx ]
  <- 13 feature bits ->  <- 2 param indices ->
```

- **Feature bits** — 1 = include this feature, 0 = drop it
- **C_idx** — index into `[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]`
- **gamma_idx** — index into `[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]`

Fitness = mean 5-fold stratified CV F1 score − 0.001 × features selected

---

## Project structure

```
Hybrid SVM-GA/
|
|-- prepare_data.py          # Downloads + merges all 4 UCI datasets, cleans, scales
|-- baseline_svm.py          # Plain RBF-SVM on all 13 features (comparison baseline)
|-- genetic_algorithm.py     # GA engine — wires into FitnessEvaluator automatically
|-- cuda_accelerator.py      # CPUEvaluator, CuMLEvaluator, CuPyKernelEvaluator, FitnessEvaluator
|-- ga_svm_trainer.py        # Decodes best chromosome, trains final model, 10-fold CV
|-- gpu_benchmark.py         # CPU vs cuML vs CuPy speedup benchmark — produces paper figure
|-- statistical_analysis.py  # 5-seed eval, Wilcoxon test, 10-fold CV, convergence analysis
|-- main.py                  # FastAPI backend
|
|-- App.jsx / Dashboard.jsx / Predict.jsx / Training.jsx / Results.jsx
|-- api.js / main.jsx / index.html / index.css / vite.config.js
|
|-- requirements.txt
`-- package.json
```

---

## Setup

### Python (CPU path — works everywhere)
```bash
pip install -r requirements.txt
```

### Python (GPU path — required for speedup results)
```bash
# NVIDIA GPU + CUDA 11.8+ required
conda install -c rapidsai cuml cupy
```

### Node
```bash
npm install
```

---

## Running the project

### Step 1 — Data pipeline
```bash
python prepare_data.py
```
Downloads and merges all 4 UCI Heart Disease sub-datasets (920 patients total). Imputes missing values with column mode, normalises with `StandardScaler`, saves stratified 80/20 split to `data/`.

Expected output:
```
[data] cleveland     — 303 rows
[data] hungarian     — 294 rows
[data] switzerland   — 123 rows
[data] va            — 200 rows
[data] Combined: 920 rows from 4 datasets
[preprocess] Train: (736, 13), Test: (184, 13)
[preprocess] Class balance (train) — 0: 329, 1: 407
```

### Step 2 — Baseline SVM
```bash
python baseline_svm.py
```
Trains plain RBF-SVM (`C=1.0`, `gamma=scale`) on all 13 features. Saves metrics, confusion matrix, and ROC curve to `results/`. Record these numbers — they are the comparison baseline.

### Step 3 — GPU Benchmark (run on GPU machine)
```bash
python gpu_benchmark.py
```
Times CPU vs cuML vs CuPy fitness evaluation across population sizes `[10, 20, 30, 60, 100]`, averaged over 3 runs each. Saves `benchmark.json` and `gpu_speedup.png` to `results/`. **This is the key figure for the paper.**

On CPU-only machines it records CPU times only and notes that GPU results require RAPIDS.

### Step 4 — GA-SVM (auto-selects GPU if available)
```bash
python genetic_algorithm.py
```
Runs the GA (default: 60 individuals x 80 generations). `FitnessEvaluator` automatically selects the best available backend:
- cuML available → uses `CuMLEvaluator` (GPU SVC)
- CuPy available → uses `CuPyKernelEvaluator` (GPU kernel)
- Neither → falls back to `CPUEvaluator` (joblib parallel)

The evaluator mode is printed at startup and logged per generation.

### Step 5 — GA-SVM final model
```bash
python ga_svm_trainer.py
```
Decodes the best chromosome, retrains the final SVM on the full training set, evaluates on the test set, and runs **10-fold stratified CV** on the full 920-sample dataset for paper-quality `mean ± std` metrics.

### Step 6 — Statistical analysis
```bash
python statistical_analysis.py
```
Runs after Steps 2-5. Performs:
- **5-seed repeated evaluation** — 5 seeds x 10-fold CV = 50 evaluations per model
- **Wilcoxon signed-rank test** — GA-SVM vs Baseline significance (p < 0.05)
- **Paired t-test** — secondary significance test
- **Convergence analysis** — plateau %, diversity collapse, convergence generation

Prints paper-ready table:
```
Metric       Baseline SVM          GA-SVM (GPU)    p-value  Sig
accuracy   0.XXXX +/- 0.XXXX   0.XXXX +/- 0.XXXX   0.XXXX   *
f1         0.XXXX +/- 0.XXXX   0.XXXX +/- 0.XXXX   0.XXXX   *
auc        0.XXXX +/- 0.XXXX   0.XXXX +/- 0.XXXX   0.XXXX
```

### Step 7 — Start the API
```bash
uvicorn main:app --reload --port 8000
```

### Step 8 — Start the frontend
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

---

## GPU acceleration details

### Three evaluator backends (cuda_accelerator.py)

| Mode | Library | How it works | When to use |
|---|---|---|---|
| `cpu` | sklearn + joblib | Parallel sklearn SVC across CPU cores | No GPU / fallback |
| `cuml` | RAPIDS cuML | Drop-in GPU SVC, data on GPU throughout | Best option with NVIDIA GPU |
| `cupy` | CuPy | RBF kernel matrix on GPU, sklearn SVC in precomputed mode | GPU without RAPIDS |

Auto-selected via `FitnessEvaluator(mode="auto")`.

### Key design: single GPU upload

```python
# Data uploaded to GPU ONCE before the GA loop starts
self._X_gpu = cp.asarray(X_train, dtype=cp.float32)
self._y_gpu = cp.asarray(y_train, dtype=cp.int32)

# Each fitness call slices the already-transferred GPU array
X_sub = self._X_gpu[:, mask]   # no transfer overhead per call
```

This is the critical optimisation — naive implementations re-upload data on every fitness call (thousands of times), which eliminates any GPU benefit.

### Evaluator mode in GAConfig

```python
cfg = GAConfig(
    pop_size=60,
    n_generations=80,
    evaluator_mode="auto"   # "auto" | "cuml" | "cupy" | "cpu"
)
```

---

## GA configuration

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 60 | Chromosomes per generation |
| `n_generations` | 80 | Generations to evolve |
| `crossover_rate` | 0.80 | Two-point crossover probability |
| `mutation_rate` | 0.02 | Per-gene bit-flip probability |
| `tournament_k` | 3 | Tournament selection pool size |
| `elitism_n` | 2 | Top N chromosomes carried over unchanged |
| `feature_penalty` | 0.001 | Sparsity penalty per selected feature |
| `min_features` | 2 | Minimum features enforced at init and mutation |
| `evaluator_mode` | `"auto"` | Fitness evaluator backend |

---

## Statistical evaluation methodology

| Method | Purpose |
|---|---|
| 10-fold stratified CV | Final reported accuracy/F1/AUC with mean ± std |
| 5-seed repeated evaluation | Robustness across random initialisations |
| Wilcoxon signed-rank test | Non-parametric significance (p < 0.05) |
| Paired t-test | Secondary parametric significance test |
| Convergence analysis | Plateau %, diversity collapse, convergence generation |
| GPU speedup benchmark | Wall-clock time CPU vs GPU across population sizes |

---

## API routes

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/meta` | Feature names, ranges, descriptions |
| GET | `/api/gpu-info` | CUDA device status and evaluator mode |
| GET | `/api/results/baseline` | Baseline SVM metrics |
| GET | `/api/results/gasvm` | GA-SVM metrics + 10-fold CV numbers |
| GET | `/api/results/comparison` | Side-by-side comparison |
| GET | `/api/results/ga-history` | GA convergence history with per-gen timing |
| POST | `/api/predict` | Predict from patient features |
| POST | `/api/train/ga` | Launch GA training with SSE progress stream |

---

## Dataset

Combined UCI Heart Disease — 920 patients, 13 features, binary target.
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

| Sub-dataset | Patients | Disease |
|---|---|---|
| Cleveland | 303 | 139 |
| Hungarian | 294 | 106 |
| Switzerland | 123 | 115 |
| VA Long Beach | 200 | 149 |
| **Combined** | **920** | **509** |

Missing values imputed with column mode. Train/test: 736 / 184 (stratified 80/20).

---

## Results

All metrics: mean ± std over 10-fold stratified CV on 920 patients.

| Metric | Baseline SVM | GA-SVM (CPU) | GA-SVM (GPU) | p-value |
|---|---|---|---|---|
| Accuracy | TBD | TBD | TBD | TBD |
| F1 Score | TBD | TBD | TBD | TBD |
| AUC-ROC | TBD | TBD | TBD | TBD |
| Features used | 13/13 | TBD/13 | TBD/13 | — |
| Time / generation | ~3-4s | ~3-4s | TBD (GPU) | — |
| Speedup | 1x | 1x | TBDx | — |

> Run Steps 1-6 on the GPU machine to populate this table.

---

## Dependencies

**Python:** scikit-learn, numpy, pandas, scipy, fastapi, uvicorn, joblib, matplotlib, seaborn, pydantic
**Node:** React 18, React Router 6, Vite 5
**GPU (required for speedup results):** RAPIDS cuML, CuPy (NVIDIA GPU + CUDA 11.8+)
