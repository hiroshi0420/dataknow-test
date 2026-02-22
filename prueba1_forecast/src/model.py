"""Modelado y forecast.

Se usa ETS (Holt-Winters Exponential Smoothing) por robustez e interpretabilidad
para series mensuales.

Notas:
- Para intervalos de confianza se usa simulación (simulate/repetitions).
- Los equipos se proyectan a partir de las materias primas proyectadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    series_name: str
    historical: pd.Series
    forecast_mean: pd.Series
    forecast_lower_80: pd.Series
    forecast_upper_80: pd.Series
    forecast_lower_95: pd.Series
    forecast_upper_95: pd.Series
    model_summary: str = ""


def fit_ets(series: pd.Series, seasonal_periods: int = 12):
    """Ajusta ETS con tendencia aditiva y estacionalidad multiplicativa."""
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="mul",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


def forecast_from_fitted(fitted_model, horizon: int, n_simulations: int = 500) -> tuple[pd.Series, dict[str, pd.Series], dict[str, pd.Series]]:
    """Forecast puntual + IC 80/95% via simulación."""
    mean_fc = fitted_model.forecast(horizon)

    simulations = fitted_model.simulate(horizon, repetitions=n_simulations)
    # simulations: (horizon x repetitions)

    lower_80 = np.percentile(simulations, 10, axis=1)
    upper_80 = np.percentile(simulations, 90, axis=1)
    lower_95 = np.percentile(simulations, 2.5, axis=1)
    upper_95 = np.percentile(simulations, 97.5, axis=1)

    idx = simulations.index

    ci80 = {
        "lower": pd.Series(lower_80, index=idx),
        "upper": pd.Series(upper_80, index=idx),
    }
    ci95 = {
        "lower": pd.Series(lower_95, index=idx),
        "upper": pd.Series(upper_95, index=idx),
    }

    return mean_fc, ci80, ci95


def forecast_series_ets(series: pd.Series, horizon: int = 36, n_simulations: int = 500) -> ForecastResult:
    fitted = fit_ets(series)
    mean_fc, ci80, ci95 = forecast_from_fitted(fitted, horizon=horizon, n_simulations=n_simulations)
    return ForecastResult(
        series_name=str(series.name),
        historical=series,
        forecast_mean=mean_fc,
        forecast_lower_80=ci80["lower"],
        forecast_upper_80=ci80["upper"],
        forecast_lower_95=ci95["lower"],
        forecast_upper_95=ci95["upper"],
        model_summary="ETS(trend=add, seasonal=mul, m=12)",
    )


def forecast_equipos_from_materials(df: pd.DataFrame, horizon: int = 36, n_simulations: int = 500) -> dict[str, ForecastResult]:
    """Proyecta X/Y/Z (ETS) y deriva equipo1/equipo2 aplicando fórmulas."""
    results: dict[str, ForecastResult] = {}

    for col in ["X", "Y", "Z"]:
        results[col] = forecast_series_ets(df[col].dropna(), horizon=horizon, n_simulations=n_simulations)

    # medios
    e1_mean = 0.20 * results["X"].forecast_mean + 0.80 * results["Y"].forecast_mean
    e2_mean = (results["X"].forecast_mean + results["Y"].forecast_mean + results["Z"].forecast_mean) / 3

    # ICs: propagación lineal (independencia; aproximación conservadora)
    e1_l95 = 0.20 * results["X"].forecast_lower_95 + 0.80 * results["Y"].forecast_lower_95
    e1_u95 = 0.20 * results["X"].forecast_upper_95 + 0.80 * results["Y"].forecast_upper_95
    e1_l80 = 0.20 * results["X"].forecast_lower_80 + 0.80 * results["Y"].forecast_lower_80
    e1_u80 = 0.20 * results["X"].forecast_upper_80 + 0.80 * results["Y"].forecast_upper_80

    e2_l95 = (results["X"].forecast_lower_95 + results["Y"].forecast_lower_95 + results["Z"].forecast_lower_95) / 3
    e2_u95 = (results["X"].forecast_upper_95 + results["Y"].forecast_upper_95 + results["Z"].forecast_upper_95) / 3
    e2_l80 = (results["X"].forecast_lower_80 + results["Y"].forecast_lower_80 + results["Z"].forecast_lower_80) / 3
    e2_u80 = (results["X"].forecast_upper_80 + results["Y"].forecast_upper_80 + results["Z"].forecast_upper_80) / 3

    results["equipo1"] = ForecastResult(
        series_name="equipo1",
        historical=df["equipo1"],
        forecast_mean=e1_mean,
        forecast_lower_80=e1_l80,
        forecast_upper_80=e1_u80,
        forecast_lower_95=e1_l95,
        forecast_upper_95=e1_u95,
        model_summary="Derivado de X/Y (0.20*X + 0.80*Y)",
    )

    results["equipo2"] = ForecastResult(
        series_name="equipo2",
        historical=df["equipo2"],
        forecast_mean=e2_mean,
        forecast_lower_80=e2_l80,
        forecast_upper_80=e2_u80,
        forecast_lower_95=e2_l95,
        forecast_upper_95=e2_u95,
        model_summary="Derivado de X/Y/Z ((X+Y+Z)/3)",
    )

    return results


def backtest_rolling_1step(series: pd.Series, test_months: int = 12) -> dict[str, float]:
    """Backtesting rolling 1-step (MAE, RMSE, MAPE) sobre los últimos N meses."""
    n = len(series)
    train_end = n - test_months
    preds: list[float] = []
    reals: list[float] = []

    for i in range(test_months):
        train = series.iloc[: train_end + i]
        fitted = fit_ets(train)
        pred = float(fitted.forecast(1).iloc[0])
        real = float(series.iloc[train_end + i])
        preds.append(pred)
        reals.append(real)

    preds_arr = np.array(preds)
    reals_arr = np.array(reals)

    err = preds_arr - reals_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / reals_arr)) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_fitted_models(df_materials: pd.DataFrame, out_dir: Path) -> Path:
    """Entrena ETS para X/Y/Z y guarda un bundle joblib para scoring."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = {
        "X": fit_ets(df_materials["X"].dropna()),
        "Y": fit_ets(df_materials["Y"].dropna()),
        "Z": fit_ets(df_materials["Z"].dropna()),
        "last_month": df_materials.index.max(),
    }
    bundle_path = out_dir / "ets_materials_bundle.joblib"
    joblib.dump(fitted, bundle_path)
    return bundle_path


def load_fitted_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def score_from_bundle(bundle: dict[str, Any], horizon: int = 36, n_simulations: int = 300) -> dict[str, dict[str, list[float]]]:
    """Genera forecast de equipos a partir del bundle entrenado (para API/AML)."""
    fitted_x = bundle["X"]
    fitted_y = bundle["Y"]
    fitted_z = bundle["Z"]

    mean_x, ci80_x, ci95_x = forecast_from_fitted(fitted_x, horizon=horizon, n_simulations=n_simulations)
    mean_y, ci80_y, ci95_y = forecast_from_fitted(fitted_y, horizon=horizon, n_simulations=n_simulations)
    mean_z, ci80_z, ci95_z = forecast_from_fitted(fitted_z, horizon=horizon, n_simulations=n_simulations)

    e1_mean = 0.20 * mean_x + 0.80 * mean_y
    e2_mean = (mean_x + mean_y + mean_z) / 3

    e1_l95 = 0.20 * ci95_x["lower"] + 0.80 * ci95_y["lower"]
    e1_u95 = 0.20 * ci95_x["upper"] + 0.80 * ci95_y["upper"]

    e2_l95 = (ci95_x["lower"] + ci95_y["lower"] + ci95_z["lower"]) / 3
    e2_u95 = (ci95_x["upper"] + ci95_y["upper"] + ci95_z["upper"]) / 3

    # serializable payload
    idx = [d.strftime("%Y-%m-%d") for d in e1_mean.index]

    return {
        "index": idx,
        "equipo1": {
            "mean": [float(v) for v in e1_mean.values],
            "lower95": [float(v) for v in e1_l95.values],
            "upper95": [float(v) for v in e1_u95.values],
        },
        "equipo2": {
            "mean": [float(v) for v in e2_mean.values],
            "lower95": [float(v) for v in e2_l95.values],
            "upper95": [float(v) for v in e2_u95.values],
        },
    }

