# CardioGA — Hybrid GA-SVM Heart Disease Classifier

A hybrid machine learning system that uses a **Genetic Algorithm (GA)** to simultaneously perform feature selection and SVM hyperparameter optimisation for heart disease prediction on the UCI Cleveland dataset. Includes a FastAPI backend and a React + Vite frontend dashboard.

---

## How it works

Plain SVM has two weaknesses on clinical data: irrelevant features add noise, and performance is highly sensitive to `C` and `γ`. This project solves both problems at once by encoding the question *"which features to use, and what values of C and γ?"* as a GA optimisation problem.

```
Chromosome = [ f1 f2 f3 ... f13 | C_idx | gamma_idx ]
               ← 13 feature bits →  ← 2 param indices →
```

Each chromosome is evaluated by training an RBF-SVM on the selected features with the decoded hyperparameters and measuring 5-fold stratified cross-validation F1 score. The GA evolves the population over generations until it converges on a high-fitness chromosome. That chromosome is then used to train the final model on the full training set.

---

## Project structure

```
Hybrid SVM-GA/
│
├── prepare_data.py        # Downloads UCI dataset, cleans, scales, saves artifacts
├── baseline_svm.py        # Trains plain SVM on all 13 features (comparison baseline)
├── genetic_algorithm.py   # Full GA engine: chromosome, fitness, selection, crossover, mutation
├── ga_svm_trainer.py      # Decodes best chromosome, trains final GA-SVM, produces plots
├── cuda_accelerator.py    # GPU fitness evaluators: CPUEvaluator, CuMLEvaluator, CuPyKernelEvaluator
├── main.py                # FastAPI backend — all API routes + SSE streaming for live training
│
├── App.jsx                # React app shell with sidebar navigation
├── Dashboard.jsx          # Overview page: metric cards, comparison table, mini convergence chart
├── Predict.jsx            # Patient input sliders → dual-model risk prediction
├── Training.jsx           # Live GA training: config sliders, real-time convergence chart, log terminal
├── Results.jsx            # Full results: metrics, confusion matrices, ROC curves, feature table
├── api.js                 # All fetch calls to the FastAPI backend, including SSE stream handler
├── main.jsx               # React entry point
├── index.html             # HTML shell
├── index.css              # Global styles
├── vite.config.js         # Vite config with /api proxy to FastAPI
│
├── requirements.txt       # Python dependencies
└── package.json           # Node dependencies
```

After running the training scripts, two additional directories are created automatically:

```
data/       # X_train.npy, X_test.npy, y_train.npy, y_test.npy, scaler.pkl, meta.json
results/    # baseline_results.json, ga_results.json, ga_svm_results.json, *.pkl, *.png
```

---

## Setup

### Python environment

```bash
conda create -n ga-svm python=3.10
conda activate ga-svm
pip install -r requirements.txt
```

**Optional GPU path** (requires NVIDIA GPU + CUDA 11.8+):
```bash
conda install -c rapidsai cuml cupy
```

### Node environment

```bash
npm install
```

---

## Running the project

Run the steps in order. Each script is self-contained and saves its outputs for the next step.

### Step 1 — Data pipeline
```bash
python prepare_data.py
```
Downloads the UCI Cleveland dataset, imputes 14 missing values in `ca` and `thal`, normalises features, and saves train/test splits to `data/`.

### Step 2 — Baseline SVM
```bash
python baseline_svm.py
```
Trains a plain RBF-SVM on all 13 features. Saves metrics, confusion matrix, and ROC curve to `results/`. This is the comparison baseline (~82–85% accuracy).

### Step 3 — Genetic Algorithm
```bash
python genetic_algorithm.py
```
Runs the GA (default: 60 individuals × 80 generations). Saves the best chromosome, full generation history, and convergence plot to `results/`.

### Step 4 — GA-SVM final model
```bash
python ga_svm_trainer.py
```
Decodes the best chromosome, retrains the final SVM on the full training set with the GA-selected features and hyperparameters, evaluates on the test set, and saves a 4-panel comparison figure.

### Step 5 — Start the API
```bash
uvicorn main:app --reload --port 8000
```

### Step 6 — Start the frontend
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

---

## GA configuration

Key parameters in `genetic_algorithm.py` → `GAConfig`:

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
- **γ**: `[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]`

---

## CUDA acceleration

`cuda_accelerator.py` provides three evaluator backends selected automatically at runtime:

| Mode | Requirement | How it works |
|---|---|---|
| `cuml` | RAPIDS cuML | Drop-in GPU SVC; data uploaded to GPU once before the GA loop |
| `cupy` | CuPy | RBF kernel matrix computed on GPU, sklearn SVC runs in precomputed-kernel mode |
| `cpu` | None | joblib-parallel sklearn SVC (default fallback) |

The evaluator is selected via `FitnessEvaluator(mode="auto")`. GPU data transfer happens once before the loop — not per fitness call.

Realistic speedup: GPU wins when population size > 50 and generations > 30 (1500+ SVM fits per run). On the small UCI dataset (303 samples), CPU with joblib parallelism is competitive below those thresholds.

---

## API routes

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/meta` | Feature names, ranges, descriptions |
| GET | `/api/gpu-info` | CUDA device status |
| GET | `/api/results/baseline` | Baseline SVM metrics |
| GET | `/api/results/gasvm` | GA-SVM metrics |
| GET | `/api/results/comparison` | Side-by-side comparison object |
| GET | `/api/results/ga-history` | GA convergence history |
| POST | `/api/predict` | Predict from patient features (both models) |
| POST | `/api/train/baseline` | Retrain baseline SVM in background |
| POST | `/api/train/ga` | Launch GA training with SSE progress stream |

---

## Frontend pages

- **Dashboard** — metric summary cards, model comparison table, GA-selected feature chips, mini convergence chart
- **Predict** — 13 clinical feature sliders with high-risk/low-risk presets, dual-model prediction output with probability bars
- **Training** — GA config sliders, live convergence chart, real-time training log terminal, best chromosome display
- **Results** — tabbed view: metrics, confusion matrices, ROC curves, feature selection table, convergence statistics

---

## Dataset

UCI Cleveland Heart Disease dataset — 303 patients, 13 features, binary target (0 = no disease, 1 = disease).  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

14 missing values in `ca` and `thal` columns are imputed with the column mode.

---

## Dependencies

**Python**: scikit-learn, numpy, pandas, fastapi, uvicorn, joblib, matplotlib, seaborn, pydantic  
**Node**: React 18, React Router 6, Vite 5  
**Optional GPU**: RAPIDS cuML, CuPy
