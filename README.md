# CardioGA — Hybrid GA-SVM/MLP Heart Disease Classifier

A hybrid machine learning system that uses a **Genetic Algorithm (GA)** to simultaneously perform feature selection and model hyperparameter optimisation for heart disease prediction on the combined UCI Heart Disease dataset (all 4 sub-datasets: Cleveland, Hungarian, Switzerland, VA Long Beach — 920 patients total).

Three models are implemented and compared:

- **Baseline SVM** — plain RBF-SVM on all 13 features, no tuning
- **GA-SVM** — GA selects features and tunes `C` and `γ` for an RBF-SVM
- **GA-MLP** — GA selects features and tunes learning rate, hidden size, depth, and dropout for a PyTorch MLP trained on GPU

Evaluation follows publication-quality standards: 10-fold stratified cross-validation for all final reported metrics, 5-seed repeated evaluation for robustness, and Wilcoxon signed-rank statistical significance testing.

---

## Why all 4 UCI sub-datasets?

The original Cleveland-only dataset has 303 patients — sufficient for SVM but too small for a neural network (roughly 82 training samples per class after splitting). To give the GA-MLP a fair evaluation, all 4 UCI Heart Disease sub-datasets are merged:

| Sub-dataset | Patients | Disease cases |
|---|---|---|
| Cleveland | 303 | 139 |
| Hungarian | 294 | 106 |
| Switzerland | 123 | 115 |
| VA Long Beach | 200 | 149 |
| **Combined** | **920** | **509** |

All 4 sub-datasets share the same 13 features and target format. Missing values (heavy in `ca`, `thal`, `slope` for non-Cleveland sets) are imputed with column mode across the full combined dataset before splitting.

---

## How it works

Plain SVM has two weaknesses on clinical data: irrelevant features add noise, and performance is highly sensitive to `C` and `γ`. This project solves both problems at once by encoding the question *"which features to use, and what hyperparameters?"* as a GA optimisation problem.

### GA-SVM chromosome (15 genes)
```
[ f1 f2 f3 ... f13 | C_idx | gamma_idx ]
  <- 13 feature bits ->  <- 2 param indices ->
```

### GA-MLP chromosome (17 genes)
```
[ f1 f2 f3 ... f13 | lr_idx | hidden_idx | dropout_idx | depth_idx ]
  <- 13 feature bits ->  <-       4 architecture indices        ->
```

Each chromosome is evaluated by training the model on the selected features with the decoded hyperparameters and measuring 5-fold stratified cross-validation F1 score minus a sparsity penalty. The GA evolves the population over generations until it converges. The best chromosome is then used to train the final model on the full training set.

For GA-MLP, each fitness call trains a PyTorch MLP for **200 epochs on GPU** across 5 folds.

**Total GA-MLP training runs:**
```
30 pop x 40 gen x 5 folds x 200 epochs = 1,200,000 epoch-equivalents
```

---

## Project structure

```
Hybrid SVM-GA/
|
|-- prepare_data.py          # Downloads + merges all 4 UCI datasets, cleans, scales, saves artifacts
|-- baseline_svm.py          # Trains plain SVM on all 13 features (comparison baseline)
|-- genetic_algorithm.py     # GA engine for SVM: chromosome, fitness, selection, crossover, mutation
|-- ga_svm_trainer.py        # Decodes best GA-SVM chromosome, trains final model, 10-fold CV
|-- ga_mlp_trainer.py        # GA engine for MLP: PyTorch GPU training inside fitness function, 10-fold CV
|-- cuda_accelerator.py      # GPU fitness evaluators: CPUEvaluator, CuMLEvaluator, CuPyKernelEvaluator
|-- statistical_analysis.py  # Fix 1+4+5: 5-seed eval, Wilcoxon test, 10-fold CV, convergence analysis
|-- main.py                  # FastAPI backend: all API routes + SSE streaming for live training
|
|-- App.jsx                  # React app shell with sidebar navigation
|-- Dashboard.jsx            # Overview: 3-model metric cards, comparison table, convergence chart
|-- Predict.jsx              # Patient input sliders -> dual-model risk prediction
|-- Training.jsx             # Live GA training: config sliders, real-time chart, log terminal
|-- Results.jsx              # Full results: metrics, confusion matrices, ROC curves, feature table
|-- api.js                   # All fetch calls to the FastAPI backend, including SSE stream handler
|-- main.jsx                 # React entry point
|-- index.html               # HTML shell
|-- index.css                # Global styles
|-- vite.config.js           # Vite config with /api proxy to FastAPI
|
|-- requirements.txt         # Python dependencies
`-- package.json             # Node dependencies
```

After running the training scripts, two directories are created automatically:

```
data/       # X_train.npy, X_test.npy, y_train.npy, y_test.npy, scaler.pkl, meta.json
results/    # baseline_results.json, ga_results.json, ga_svm_results.json,
            # ga_mlp_results.json, statistical_report.json, *.pkl, *.pt, *.png
```

---

## Setup

### Python environment

```bash
pip install -r requirements.txt
```

**GPU path for GA-MLP** (requires NVIDIA GPU + CUDA 11.8+):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Optional GPU path for GA-SVM** (requires RAPIDS):
```bash
conda install -c rapidsai cuml cupy
```

### Node environment

```bash
npm install
```

---

## Running the project

Run the steps in order. Each script saves its outputs for the next step.

### Step 1 — Data pipeline
```bash
python prepare_data.py
```
Downloads and merges all 4 UCI Heart Disease sub-datasets into a single 920-patient dataset. Imputes missing values across all columns with column mode, normalises all features with `StandardScaler`, and saves a stratified 80/20 train/test split to `data/`.

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
Trains a plain RBF-SVM (`C=1.0`, `gamma=scale`) on all 13 features using the full 736-sample training set. Evaluates on the 184-sample test set. Saves accuracy, F1, AUC-ROC, confusion matrix PNG, and ROC curve PNG to `results/`. This is the comparison baseline — record these numbers for the paper.

### Step 3 — GA-SVM
```bash
python genetic_algorithm.py
```
Runs the GA (default: 60 individuals x 80 generations) to find the optimal feature subset and SVM hyperparameters. Each fitness call runs 5-fold stratified CV on the training set only — test data is never seen during the GA loop. Saves the best chromosome, full generation history, and convergence plot to `results/`. Takes ~4-10 minutes on CPU.

### Step 4 — GA-SVM final model
```bash
python ga_svm_trainer.py
```
Decodes the best chromosome from Step 3, retrains the final SVM on the full 736-sample training set, evaluates on the 184-sample test set, and runs **10-fold stratified CV** on the full 920-sample dataset for paper-quality `mean ± std` metrics. Saves a 4-panel comparison figure.

### Step 5 — GA-MLP (GPU required)
```bash
python ga_mlp_trainer.py
```
Runs the GA (default: 30 individuals x 40 generations) where each fitness call trains a PyTorch MLP for 200 epochs on GPU across 5 CV folds. After the GA converges, trains the final model and runs **10-fold CV** for paper-quality metrics. Saves model weights (`.pt`), convergence plot, and 3-model comparison figure.

**This step requires an NVIDIA GPU with CUDA 11.8+. Do not run on CPU — it will take days.**

When the GPU is detected, the script prints:
```
[device] Using: cuda
[device] GPU: NVIDIA GeForce RTX XXXX
```

Expected runtime: ~4 hours on RTX 3090 / A100.

### Step 6 — Statistical analysis
```bash
python statistical_analysis.py
```
Runs after all training scripts are complete. Performs:

- **5-seed repeated evaluation** — runs each model 5 times with different random seeds (42, 7, 13, 99, 2024), each with 10-fold CV = 50 evaluations per model
- **Wilcoxon signed-rank test** — tests whether GA-SVM and GA-MLP are statistically significantly better than baseline (p < 0.05)
- **Paired t-test** — secondary significance test
- **Convergence analysis** — plateau detection, diversity collapse, convergence generation, per-generation improvement
- **Box plot summary** — visual comparison of score distributions across all seeds

Prints a paper-ready results table:
```
Metric       Baseline SVM          GA-SVM          p-value  Sig
accuracy   0.XXXX +/- 0.XXXX   0.XXXX +/- 0.XXXX   0.XXXX   *
f1         0.XXXX +/- 0.XXXX   0.XXXX +/- 0.XXXX   0.XXXX   *
auc        0.XXXX +/- 0.XXXX   0.XXXX +/- 0.XXXX   0.XXXX
```

Saves `statistical_report.json`, `statistical_summary.png`, and `ga_svm_convergence_analysis.png` to `results/`.

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

## GA-SVM configuration

Key parameters in `genetic_algorithm.py` -> `GAConfig`:

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 60 | Number of chromosomes per generation |
| `n_generations` | 80 | Number of generations to evolve |
| `crossover_rate` | 0.80 | Probability of two-point crossover |
| `mutation_rate` | 0.02 | Per-gene bit-flip probability |
| `tournament_k` | 3 | Tournament selection pool size |
| `elitism_n` | 2 | Top N chromosomes carried over unchanged |
| `feature_penalty` | 0.001 | Sparsity penalty per selected feature |
| `min_features` | 2 | Minimum features enforced at init and mutation |

Hyperparameter search grids:
- **C**: `[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]`
- **gamma**: `[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]`

---

## GA-MLP configuration

Key parameters in `ga_mlp_trainer.py` -> `GAMLPConfig`:

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 30 | Number of chromosomes per generation |
| `n_generations` | 40 | Number of generations to evolve |
| `crossover_rate` | 0.80 | Probability of two-point crossover |
| `mutation_rate` | 0.02 | Per-gene bit-flip probability |
| `tournament_k` | 3 | Tournament selection pool size |
| `elitism_n` | 2 | Top N chromosomes carried over unchanged |

MLP hyperparameter search grids:
- **Learning rate**: `[1e-4, 5e-4, 1e-3, 5e-3, 1e-2]`
- **Hidden size**: `[32, 64, 128, 256, 512]`
- **Dropout**: `[0.0, 0.1, 0.2, 0.3, 0.4]`
- **Depth**: `[1, 2, 3, 4]` hidden layers

Each MLP uses `BatchNorm -> ReLU -> Dropout` per layer, `BCEWithLogitsLoss`, Adam optimiser with `weight_decay=1e-4`, and cosine annealing LR scheduler over 200 epochs.

---

## Statistical evaluation methodology

All final reported metrics follow publication-quality standards:

| Method | Purpose |
|---|---|
| 10-fold stratified CV | Final reported accuracy/F1/AUC with mean ± std |
| 5-seed repeated evaluation | Robustness check across random initialisations |
| Wilcoxon signed-rank test | Non-parametric significance test (p < 0.05 threshold) |
| Paired t-test | Secondary parametric significance test |
| Convergence analysis | Plateau %, diversity collapse, convergence generation |

The `statistical_analysis.py` script produces `statistical_report.json` containing all of the above, ready to be cited directly in the paper.

---

## CUDA acceleration

### GA-SVM (cuda_accelerator.py)

Three evaluator backends selected automatically at runtime:

| Mode | Requirement | How it works |
|---|---|---|
| `cuml` | RAPIDS cuML | Drop-in GPU SVC; data uploaded to GPU once before the GA loop |
| `cupy` | CuPy | RBF kernel matrix computed on GPU, sklearn SVC in precomputed-kernel mode |
| `cpu` | None | joblib-parallel sklearn SVC (default fallback) |

Selected via `FitnessEvaluator(mode="auto")`. GPU data transfer happens once before the loop, not per fitness call.

### GA-MLP (ga_mlp_trainer.py)

PyTorch tensors are moved to `DEVICE` (auto-detected `cuda` or `cpu`) inside each fitness call:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tr = torch.tensor(X_sub[tr_idx]).to(DEVICE)
```
The MLP model, optimiser, and all forward/backward passes run entirely on GPU.

---

## API routes

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/meta` | Feature names, ranges, descriptions |
| GET | `/api/gpu-info` | CUDA device status |
| GET | `/api/results/baseline` | Baseline SVM metrics |
| GET | `/api/results/gasvm` | GA-SVM metrics |
| GET | `/api/results/gamlp` | GA-MLP metrics |
| GET | `/api/results/comparison` | Side-by-side comparison (all 3 models) |
| GET | `/api/results/ga-history` | GA-SVM convergence history |
| GET | `/api/results/gamlp-history` | GA-MLP convergence history |
| POST | `/api/predict` | Predict from patient features (baseline + GA-SVM) |
| POST | `/api/train/baseline` | Retrain baseline SVM in background |
| POST | `/api/train/ga` | Launch GA-SVM training with SSE progress stream |

---

## Frontend pages

- **Dashboard** — 4 metric cards (GA-SVM + GA-MLP), 3-model comparison table showing accuracy/F1/AUC/features/device, GA-selected feature chips, mini convergence chart
- **Predict** — 13 clinical feature sliders with high-risk/low-risk presets, dual-model prediction output with probability bars
- **Training** — GA config sliders, live SSE-driven convergence chart, real-time training log terminal, best chromosome display
- **Results** — tabbed view: metrics table (all 3 models with winner column), confusion matrices, ROC curves (all 3 overlaid), feature selection table, convergence statistics

---

## Dataset

Combined UCI Heart Disease dataset — 920 patients, 13 features, binary target (0 = no disease, 1 = disease).
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

Four sub-datasets merged: Cleveland (303), Hungarian (294), Switzerland (123), VA Long Beach (200).
Missing values imputed with column mode across the full combined dataset before splitting.

Train/test split: 736 train / 184 test (stratified 80/20).

---

## Results

All metrics reported as mean ± std over 10-fold stratified CV on the full 920-patient dataset.

| Metric | Baseline SVM | GA-SVM | GA-MLP | p-value (GA-SVM vs Baseline) |
|---|---|---|---|---|
| Accuracy | TBD | TBD | run on GPU | TBD |
| F1 Score | TBD | TBD | run on GPU | TBD |
| AUC-ROC | TBD | TBD | run on GPU | TBD |
| Features used | 13/13 | TBD/13 | TBD/13 | — |
| Train time | ~0.1s | ~10 min (GA) | ~4 hours (GPU) | — |
| Device | CPU | CPU | CUDA | — |

> Run Steps 1-6 on the GPU machine to populate this table. The `statistical_analysis.py` script prints the completed table directly.

---

## Dependencies

**Python:** scikit-learn, numpy, pandas, scipy, fastapi, uvicorn, joblib, matplotlib, seaborn, pydantic, torch
**Node:** React 18, React Router 6, Vite 5
**Optional GPU (GA-SVM):** RAPIDS cuML, CuPy
**Required GPU (GA-MLP):** PyTorch with CUDA 11.8+
