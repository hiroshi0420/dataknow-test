"""API de inferencia local.

Sirve forecast a partir de un bundle ETS entrenado previamente por el pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .model import load_fitted_bundle, score_from_bundle

app = FastAPI(title="Cost Forecast API", version="1.0.0")


def _bundle_path() -> Path:
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    return output_dir / "models" / "ets_materials_bundle.joblib"


@lru_cache(maxsize=1)
def _load_bundle_cached():
    path = _bundle_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el bundle de modelos en {path}. Ejecuta primero el pipeline batch (main.py)."
        )
    return load_fitted_bundle(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def forecast(horizon: int = 36):
    try:
        bundle = _load_bundle_cached()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if horizon < 1 or horizon > 120:
        raise HTTPException(status_code=400, detail="horizon debe estar entre 1 y 120")

    payload = score_from_bundle(bundle, horizon=horizon)
    return payload

