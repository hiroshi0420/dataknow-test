"""Generación de artefactos (CSV + gráficas)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Docker/CI)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import ForecastResult


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_stats(stats: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "estadisticas_descriptivas.csv"
    stats.to_csv(path)
    return path


def save_history(df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "costos_historico.csv"
    df.to_csv(path)
    return path


def save_forecast_long(df_forecast: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "forecast_36m.csv"
    df_forecast.to_csv(path, index=False)
    return path


def plot_materias_primas(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Precios históricos (mensual) – Materias primas", fontsize=14)

    for ax, col in zip(axes, ["X", "Y", "Z"]):
        ax.plot(df.index, df[col], linewidth=1.5, label=col)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.xticks(rotation=45)
    plt.tight_layout()

    path = output_dir / "01_materias_primas_historico.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equipos_historico(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df["equipo1"], linewidth=2, label="Equipo 1")
    ax.plot(df.index, df["equipo2"], linewidth=2, linestyle="--", label="Equipo 2")

    ax.set_title("Costo histórico (mensual) – Equipo 1 vs Equipo 2")
    ax.set_ylabel("Precio")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = output_dir / "02_equipos_historico.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forecast(result: ForecastResult, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))

    hist = result.historical.iloc[-36:]
    ax.plot(hist.index, hist.values, linewidth=2, label="Histórico")

    cut = result.historical.index[-1]
    ax.axvline(x=cut, linestyle=":", alpha=0.7)

    fc = result.forecast_mean
    ax.plot(fc.index, fc.values, linewidth=2.5, linestyle="--", label="Forecast (media)")

    # fill_between es más estable con números de fecha
    x_fc = mdates.date2num(fc.index.to_pydatetime())

    ax.fill_between(
        x_fc,
        np.asarray(result.forecast_lower_95.values, dtype=float),
        np.asarray(result.forecast_upper_95.values, dtype=float),
        alpha=0.2,
        label="IC 95%",
    )
    ax.fill_between(
        x_fc,
        np.asarray(result.forecast_lower_80.values, dtype=float),
        np.asarray(result.forecast_upper_80.values, dtype=float),
        alpha=0.35,
        label="IC 80%",
    )

    ax.set_title(f"Forecast – {result.series_name}")
    ax.set_ylabel("Precio")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = output_dir / f"03_forecast_{result.series_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_volatility(df: pd.DataFrame, output_dir: Path) -> Path:
    pct = df[["equipo1", "equipo2"]].pct_change() * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Volatilidad mensual (% cambio)")

    for ax, col in zip(axes, ["equipo1", "equipo2"]):
        ax.bar(pct.index, pct[col], alpha=0.7, width=20)
        ax.axhline(0, linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    path = output_dir / "04_volatilidad_mensual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

