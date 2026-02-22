"""Comparación de modelos para forecasting.

Se comparan modelos sobre las series objetivo (equipo1/equipo2) usando
backtesting rolling 1-step en los últimos N meses.

Modelos soportados:
- ets: Exponential Smoothing (Holt-Winters)
- sarima: SARIMAX estacional (m=12)
- prophet: opcional (si el paquete `prophet` está instalado)

La comparación produce:
- leaderboard (métricas por serie/modelo)
- forecast 36m por modelo y selección del mejor por serie
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .model import ForecastResult, forecast_from_fitted, fit_ets


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 12) -> float:
    if len(y_train) <= m:
        scale = np.mean(np.abs(np.diff(y_train)))
    else:
        scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    scale = float(scale) if scale and not np.isnan(scale) else 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / np.where(y_true == 0, 1.0, y_true))) * 100.0)
    smape = _smape(y_true, y_pred)
    mase = _mase(y_true, y_pred, y_train=y_train, m=12)
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "mase": mase}


def _fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def _forecast_sarima(fitted, horizon: int) -> tuple[pd.Series, dict[str, pd.Series], dict[str, pd.Series]]:
    fc = fitted.get_forecast(steps=horizon)
    mean_fc = fc.predicted_mean

    ci95_df = fc.conf_int(alpha=0.05)
    ci80_df = fc.conf_int(alpha=0.20)

    # statsmodels usa nombres como lower y upper por variable
    lower95 = ci95_df.iloc[:, 0]
    upper95 = ci95_df.iloc[:, 1]
    lower80 = ci80_df.iloc[:, 0]
    upper80 = ci80_df.iloc[:, 1]

    ci80 = {"lower": lower80, "upper": upper80}
    ci95 = {"lower": lower95, "upper": upper95}
    return mean_fc, ci80, ci95


def forecast_series(series: pd.Series, model_name: str, horizon: int, n_simulations: int = 500) -> ForecastResult:
    model_name = model_name.lower()

    if model_name == "ets":
        fitted = fit_ets(series)
        mean_fc, ci80, ci95 = forecast_from_fitted(fitted, horizon=horizon, n_simulations=n_simulations)
        summary = "ETS(trend=add, seasonal=mul, m=12)"
    elif model_name == "sarima":
        fitted = _fit_sarima(series)
        mean_fc, ci80, ci95 = _forecast_sarima(fitted, horizon=horizon)
        summary = "SARIMA(order=(1,1,1), seasonal=(1,1,1,12))"
    elif model_name == "prophet":
        try:
            from prophet import Prophet  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Prophet no está instalado. Instala requirements-prophet.txt") from e

        # Fit con interval_width 0.95 para IC 95%
        dfp = series.reset_index().rename(columns={"month": "ds", series.name: "y"})
        m95 = Prophet(interval_width=0.95, yearly_seasonality=True)
        m95.fit(dfp)
        fut = m95.make_future_dataframe(periods=horizon, freq="M")
        pred95 = m95.predict(fut).tail(horizon).set_index("ds")
        mean_fc = pred95["yhat"].rename(series.name)
        ci95 = {"lower": pred95["yhat_lower"], "upper": pred95["yhat_upper"]}

        # Fit adicional para IC 80%
        m80 = Prophet(interval_width=0.80, yearly_seasonality=True)
        m80.fit(dfp)
        pred80 = m80.predict(fut).tail(horizon).set_index("ds")
        ci80 = {"lower": pred80["yhat_lower"], "upper": pred80["yhat_upper"]}
        summary = "Prophet(yearly_seasonality=True)"
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return ForecastResult(
        series_name=str(series.name),
        historical=series,
        forecast_mean=mean_fc,
        forecast_lower_80=ci80["lower"],
        forecast_upper_80=ci80["upper"],
        forecast_lower_95=ci95["lower"],
        forecast_upper_95=ci95["upper"],
        model_summary=summary,
    )


def backtest_rolling_1step_generic(
    series: pd.Series,
    model_name: str,
    test_months: int = 12,
) -> dict[str, float]:
    """Backtesting rolling 1-step para un modelo específico."""

    n = len(series)
    train_end = n - test_months

    preds: list[float] = []
    reals: list[float] = []

    for i in range(test_months):
        train = series.iloc[: train_end + i]

        if model_name == "ets":
            fitted = fit_ets(train)
            pred = float(fitted.forecast(1).iloc[0])
        elif model_name == "sarima":
            fitted = _fit_sarima(train)
            pred = float(fitted.get_forecast(steps=1).predicted_mean.iloc[0])
        elif model_name == "prophet":
            from prophet import Prophet  # type: ignore

            dfp = train.reset_index().rename(columns={"month": "ds", train.name: "y"})
            m = Prophet(interval_width=0.95, yearly_seasonality=True)
            m.fit(dfp)
            fut = m.make_future_dataframe(periods=1, freq="M")
            pred_df = m.predict(fut).tail(1)
            pred = float(pred_df["yhat"].iloc[0])
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        real = float(series.iloc[train_end + i])
        preds.append(pred)
        reals.append(real)

    y_true = np.array(reals, dtype=float)
    y_pred = np.array(preds, dtype=float)
    y_train = np.array(series.iloc[:train_end].values, dtype=float)
    return _metrics(y_true=y_true, y_pred=y_pred, y_train=y_train)


def compare_and_forecast_equipos(
    df_equipos: pd.DataFrame,
    horizon: int,
    test_months: int,
    n_simulations: int,
    models: list[str],
) -> dict[str, object]:
    """Genera leaderboard y forecast para cada modelo sobre equipo1/equipo2."""

    rows: list[dict[str, object]] = []
    forecasts: dict[str, dict[str, ForecastResult]] = {"equipo1": {}, "equipo2": {}}

    for serie in ["equipo1", "equipo2"]:
        s = df_equipos[serie].dropna()

        for model in models:
            model = model.lower()
            try:
                met = backtest_rolling_1step_generic(s, model_name=model, test_months=test_months)
                fr = forecast_series(s, model_name=model, horizon=horizon, n_simulations=n_simulations)
                forecasts[serie][model] = fr
                rows.append({"serie": serie, "modelo": model, **met, "model_summary": fr.model_summary})
            except Exception as e:  # noqa: BLE001
                # Si Prophet no está disponible u otro fallo, reporta y continúa.
                rows.append({"serie": serie, "modelo": model, "error": str(e)})

    leaderboard = pd.DataFrame(rows)

    # Selección del mejor (menor MASE, luego RMSE)
    def _pick_best(df_: pd.DataFrame) -> str:
        dfm = df_.dropna(subset=["mase", "rmse"], how="any")
        if dfm.empty:
            return "ets"
        dfm = dfm.sort_values(["mase", "rmse"]).reset_index(drop=True)
        return str(dfm.loc[0, "modelo"])

    best = {
        "equipo1": _pick_best(leaderboard[leaderboard["serie"] == "equipo1"]),
        "equipo2": _pick_best(leaderboard[leaderboard["serie"] == "equipo2"]),
    }

    if not leaderboard.empty and "modelo" in leaderboard.columns and "serie" in leaderboard.columns:
        leaderboard["is_best"] = leaderboard.apply(lambda r: str(r.get("modelo")) == best.get(str(r.get("serie"))), axis=1)

    return {"leaderboard": leaderboard, "forecasts": forecasts, "best": best}
