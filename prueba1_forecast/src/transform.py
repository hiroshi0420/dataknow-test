"""Transformaciones de negocio.

- Equipo 1 = 0.20 * X + 0.80 * Y
- Equipo 2 = (X + Y + Z) / 3

Incluye estadísticas descriptivas para reporte.
"""

from __future__ import annotations

import pandas as pd


def compute_equipment_costs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["equipo1"] = 0.20 * out["X"] + 0.80 * out["Y"]
    out["equipo2"] = (out["X"] + out["Y"] + out["Z"]) / 3
    return out


def describe_series(df: pd.DataFrame) -> pd.DataFrame:
    stats: dict[str, dict[str, float]] = {}

    for col in df.columns:
        s = df[col].dropna()
        stats[col] = {
            "Media": round(float(s.mean()), 2),
            "Mediana": round(float(s.median()), 2),
            "Desv. Estándar": round(float(s.std()), 2),
            "Mínimo": round(float(s.min()), 2),
            "Máximo": round(float(s.max()), 2),
            "Coef. Variación (%)": round(float((s.std() / s.mean()) * 100), 1),
            "Var. Total (%)": round(float(((s.iloc[-1] / s.iloc[0]) - 1) * 100), 1),
        }

    return pd.DataFrame(stats).T

