"""Orquestación del pipeline (batch)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .ingest import load_all
from .transform import compute_equipment_costs, describe_series
from .model import (
    ForecastResult,
    backtest_rolling_1step,
    forecast_equipos_from_materials,
    save_fitted_models,
)
from .report import (
    ensure_output_dir,
    plot_equipos_historico,
    plot_forecast,
    plot_materias_primas,
    plot_volatility,
    save_forecast_long,
    save_history,
    save_stats,
)


def results_to_long_df(results: dict[str, ForecastResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, res in results.items():
        for date, val in res.historical.items():
            rows.append(
                {
                    "month": date,
                    "serie": name,
                    "tipo": "historical",
                    "valor": float(val),
                    "lower_80": float("nan"),
                    "upper_80": float("nan"),
                    "lower_95": float("nan"),
                    "upper_95": float("nan"),
                }
            )

        for date in res.forecast_mean.index:
            rows.append(
                {
                    "month": date,
                    "serie": name,
                    "tipo": "forecast",
                    "valor": float(res.forecast_mean.loc[date]),
                    "lower_80": float(res.forecast_lower_80.loc[date]),
                    "upper_80": float(res.forecast_upper_80.loc[date]),
                    "lower_95": float(res.forecast_lower_95.loc[date]),
                    "upper_95": float(res.forecast_upper_95.loc[date]),
                }
            )

    return pd.DataFrame(rows)


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    horizon: int = 36,
    no_forecast: bool = False,
    n_simulations: int = 500,
) -> dict:
    """Ejecuta el pipeline end-to-end y guarda artefactos."""

    ensure_output_dir(output_dir)

    df_materials = load_all(data_dir)
    df = compute_equipment_costs(df_materials)

    stats = describe_series(df)
    save_stats(stats, output_dir)

    # Visualizaciones de histórico
    plot_materias_primas(df, output_dir)
    plot_equipos_historico(df, output_dir)
    plot_volatility(df, output_dir)

    save_history(df[["X", "Y", "Z", "equipo1", "equipo2"]], output_dir)

    artifacts: dict = {
        "months": len(df),
        "range": (df.index.min(), df.index.max()),
        "stats": stats,
    }

    if no_forecast:
        return artifacts

    # Backtesting (últimos 12 meses) sobre equipos
    bt_e1 = backtest_rolling_1step(df["equipo1"], test_months=12)
    bt_e2 = backtest_rolling_1step(df["equipo2"], test_months=12)

    # Forecast
    results = forecast_equipos_from_materials(df, horizon=horizon, n_simulations=n_simulations)

    # Plots forecast por equipo
    plot_forecast(results["equipo1"], output_dir)
    plot_forecast(results["equipo2"], output_dir)

    df_long = results_to_long_df({"equipo1": results["equipo1"], "equipo2": results["equipo2"]})
    save_forecast_long(df_long, output_dir)

    # Guarda bundle para scoring (API / Azure ML)
    models_dir = output_dir / "models"
    bundle_path = save_fitted_models(df_materials, models_dir)

    artifacts.update(
        {
            "backtest": {"equipo1": bt_e1, "equipo2": bt_e2},
            "forecast": results,
            "bundle_path": bundle_path,
        }
    )

    return artifacts


def get_default_paths() -> tuple[Path, Path]:
    data_dir = Path(os.getenv("DATA_DIR", "Datos"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    return data_dir, output_dir

