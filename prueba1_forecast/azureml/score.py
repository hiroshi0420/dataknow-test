import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# Azure ML expects init() and run()
_model_bundle = None


def _forecast_from_fitted(fitted_model, horizon: int, n_simulations: int = 200):
    mean_fc = fitted_model.forecast(horizon)
    sims = fitted_model.simulate(horizon, repetitions=n_simulations)

    l95 = np.percentile(sims, 2.5, axis=1)
    u95 = np.percentile(sims, 97.5, axis=1)

    return mean_fc, pd.Series(l95, index=sims.index), pd.Series(u95, index=sims.index)


def init():
    global _model_bundle
    # Model is mounted by Azure ML under AZUREML_MODEL_DIR
    model_dir = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    bundle_path = model_dir / "ets_materials_bundle.joblib"
    _model_bundle = joblib.load(bundle_path)


def run(raw_data):
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        horizon = int(data.get("horizon", 36))
        horizon = max(1, min(horizon, 120))

        fx = _model_bundle["X"]
        fy = _model_bundle["Y"]
        fz = _model_bundle["Z"]

        mx, lx, ux = _forecast_from_fitted(fx, horizon)
        my, ly, uy = _forecast_from_fitted(fy, horizon)
        mz, lz, uz = _forecast_from_fitted(fz, horizon)

        e1_mean = 0.20 * mx + 0.80 * my
        e2_mean = (mx + my + mz) / 3

        e1_l95 = 0.20 * lx + 0.80 * ly
        e1_u95 = 0.20 * ux + 0.80 * uy

        e2_l95 = (lx + ly + lz) / 3
        e2_u95 = (ux + uy + uz) / 3

        idx = [d.strftime("%Y-%m-%d") for d in e1_mean.index]

        return {
            "index": idx,
            "equipo1": {"mean": e1_mean.tolist(), "lower95": e1_l95.tolist(), "upper95": e1_u95.tolist()},
            "equipo2": {"mean": e2_mean.tolist(), "lower95": e2_l95.tolist(), "upper95": e2_u95.tolist()},
        }
    except Exception as e:
        return {"error": str(e)}
