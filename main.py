"""
Day 7 — FastAPI Backend
Serves predictions, training triggers, and all result data to the React frontend.

Routes:
  GET  /api/health               — liveness check
  GET  /api/meta                 — feature names, ranges, descriptions
  GET  /api/results/baseline     — baseline SVM metrics
  GET  /api/results/gasvm        — GA-SVM metrics
  GET  /api/results/comparison   — side-by-side comparison object
  GET  /api/results/ga-history   — GA convergence history
  POST /api/predict               — predict from patient features
  POST /api/train/baseline       — (re-)train baseline SVM
  POST /api/train/ga             — launch GA training run (SSE streaming)
  GET  /api/gpu-info             — CUDA device information

Run:  uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
import json
import pickle
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

log = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

app = FastAPI(title="GA-SVM Heart Disease API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}. "
                            "Run the training scripts first.")
    with open(path) as f:
        return json.load(f)


def load_pickle(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {path.name}.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    age:      float = Field(..., ge=29,  le=77,   description="Age in years")
    sex:      float = Field(..., ge=0,   le=1)
    cp:       float = Field(..., ge=0,   le=3,    description="Chest pain type")
    trestbps: float = Field(..., ge=94,  le=200,  description="Resting blood pressure")
    chol:     float = Field(..., ge=126, le=564,  description="Serum cholesterol")
    fbs:      float = Field(..., ge=0,   le=1,    description="Fasting blood sugar > 120mg/dl")
    restecg:  float = Field(..., ge=0,   le=2,    description="Resting ECG results")
    thalach:  float = Field(..., ge=71,  le=202,  description="Max heart rate achieved")
    exang:    float = Field(..., ge=0,   le=1,    description="Exercise-induced angina")
    oldpeak:  float = Field(..., ge=0.0, le=6.2,  description="ST depression")
    slope:    float = Field(..., ge=0,   le=2)
    ca:       float = Field(..., ge=0,   le=4,    description="Major vessels count")
    thal:     float = Field(..., ge=0,   le=3,    description="Thalassemia type")


class GATrainConfig(BaseModel):
    pop_size:      int   = Field(60, ge=10, le=200)
    n_generations: int   = Field(80, ge=10, le=300)
    mutation_rate: float = Field(0.02, ge=0.001, le=0.2)
    crossover_rate: float = Field(0.80, ge=0.5, le=1.0)
    use_gpu:       bool  = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/meta")
def get_meta():
    return load_json(DATA_DIR / "meta.json")


@app.get("/api/gpu-info")
def gpu_info():
    try:
        from backend.models.cuda_accelerator import GPU_INFO
        return GPU_INFO
    except Exception:
        return {"cuda_available": False, "cuml_available": False,
                "cupy_available": False, "device_name": None}


@app.get("/api/results/baseline")
def get_baseline():
    return load_json(RESULTS_DIR / "baseline_results.json")


@app.get("/api/results/gasvm")
def get_gasvm():
    return load_json(RESULTS_DIR / "ga_svm_results.json")


@app.get("/api/results/ga-history")
def get_ga_history():
    results = load_json(RESULTS_DIR / "ga_results.json")
    return {"history": results.get("history", []),
            "best_chromosome": results.get("best_chromosome", {}),
            "config": results.get("config", {})}


@app.get("/api/results/comparison")
def get_comparison():
    baseline = load_json(RESULTS_DIR / "baseline_results.json")
    gasvm    = load_json(RESULTS_DIR / "ga_svm_results.json")
    ga_res   = load_json(RESULTS_DIR / "ga_results.json")
    meta     = load_json(DATA_DIR    / "meta.json")

    best_chr = ga_res.get("best_chromosome", {})
    history  = ga_res.get("history", [])

    return {
        "baseline": {
            "accuracy":      baseline.get("accuracy"),
            "f1":            baseline.get("f1"),
            "auc":           baseline.get("auc"),
            "n_features":    baseline.get("n_features_used", 13),
            "train_time_s":  baseline.get("train_time_s"),
            "cm":            baseline.get("cm"),
            "roc":           baseline.get("roc"),
        },
        "gasvm": {
            "accuracy":      gasvm.get("accuracy"),
            "f1":            gasvm.get("f1"),
            "auc":           gasvm.get("auc"),
            "n_features":    gasvm.get("n_features_used"),
            "train_time_s":  gasvm.get("train_time_s"),
            "cm":            gasvm.get("cm"),
            "roc":           gasvm.get("roc"),
            "feature_mask":  gasvm.get("feature_mask"),
            "C":             gasvm.get("C"),
            "gamma":         gasvm.get("gamma"),
        },
        "ga_history":    history,
        "feature_names": meta.get("feature_names", []),
        "best_chromosome": best_chr,
    }


@app.post("/api/predict")
def predict(patient: PatientFeatures):
    """
    Runs both baseline SVM and GA-SVM on the given patient features.
    Returns prediction, probability, and the features each model used.
    """
    meta     = load_json(DATA_DIR / "meta.json")
    scaler   = load_pickle(DATA_DIR / "scaler.pkl")
    bl_model = load_pickle(RESULTS_DIR / "baseline_model.pkl")
    ga_bundle = load_pickle(RESULTS_DIR / "ga_svm_model.pkl")

    feature_names = meta["feature_names"]
    raw = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]])

    X_scaled = scaler.transform(raw)

    # Baseline prediction (all 13 features)
    bl_prob    = bl_model.predict_proba(X_scaled)[0, 1]
    bl_pred    = int(bl_model.predict(X_scaled)[0])

    # GA-SVM prediction (selected features only)
    mask       = ga_bundle["mask"]
    ga_model   = ga_bundle["model"]
    X_ga       = X_scaled[:, mask]
    ga_prob    = ga_model.predict_proba(X_ga)[0, 1]
    ga_pred    = int(ga_model.predict(X_ga)[0])

    selected_features = [feature_names[i] for i in range(len(mask)) if mask[i]]

    return {
        "baseline": {
            "prediction":   bl_pred,
            "probability":  round(float(bl_prob), 4),
            "label":        "High Risk" if bl_pred == 1 else "Low Risk",
            "features_used": feature_names,
        },
        "gasvm": {
            "prediction":   ga_pred,
            "probability":  round(float(ga_prob), 4),
            "label":        "High Risk" if ga_pred == 1 else "Low Risk",
            "features_used": selected_features,
            "feature_mask":  mask.astype(int).tolist(),
            "C":             ga_bundle["C"],
            "gamma":         ga_bundle["gamma"],
        },
        "consensus": ga_pred == bl_pred,
        "risk_score": round(float((bl_prob + ga_prob) / 2), 4),
    }


@app.post("/api/train/baseline")
async def train_baseline_api(background_tasks: BackgroundTasks):
    """Trigger baseline SVM training in the background."""
    def _run():
        import subprocess, sys
        subprocess.run([sys.executable, "-m",
                        "backend.models.baseline_svm"], check=True)
    background_tasks.add_task(_run)
    return {"status": "started", "message": "Baseline SVM training launched."}


@app.post("/api/train/ga")
async def train_ga_api(cfg: GATrainConfig):
    """
    Streams GA training progress via Server-Sent Events (SSE).
    Frontend receives live generation updates.
    """
    async def event_stream():
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from backend.models.genetic_algorithm import GeneticAlgorithm, GAConfig

        X_train = np.load(DATA_DIR / "X_train.npy")
        y_train = np.load(DATA_DIR / "y_train.npy")

        ga_cfg = GAConfig(
            pop_size      = cfg.pop_size,
            n_generations = cfg.n_generations,
            mutation_rate = cfg.mutation_rate,
            crossover_rate= cfg.crossover_rate,
        )
        ga = GeneticAlgorithm(ga_cfg)

        queue = asyncio.Queue()

        def progress_cb(gen, stats, best):
            payload = {
                "generation":      gen,
                "best_fitness":    round(stats.best_fitness, 6),
                "avg_fitness":     round(stats.avg_fitness, 6),
                "best_n_features": stats.best_n_features,
                "best_chromosome": best.to_dict() if best else None,
            }
            asyncio.get_event_loop().call_soon_threadsafe(
                queue.put_nowait, json.dumps(payload)
            )

        import concurrent.futures
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        future = loop.run_in_executor(
            executor, lambda: ga.run(X_train, y_train, progress_callback=progress_cb)
        )

        while not future.done():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=2.0)
                yield f"data: {msg}\n\n"
            except asyncio.TimeoutError:
                yield "data: {\"ping\": true}\n\n"

        # Drain any remaining messages
        while not queue.empty():
            msg = await queue.get()
            yield f"data: {msg}\n\n"

        best = future.result()
        yield f"data: {json.dumps({'done': True, 'best': best.to_dict()})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
