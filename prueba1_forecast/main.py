"""Batch runner.

Ejecuta:
- Ingesta y normalización de X/Y/Z
- Cálculo de costos históricos
- Estadísticas descriptivas
- Forecast ETS a 36 meses con intervalos (80/95)
- Export de CSV + gráficas + bundle de modelos

Uso:
  python main.py
  python main.py --horizon 36
  python main.py --no-forecast
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.index import get_default_paths, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost Forecast – Time Series")
    p.add_argument("--horizon", type=int, default=36, help="Meses a proyectar")
    p.add_argument("--no-forecast", action="store_true", help="Solo histórico y EDA")
    p.add_argument("--n-simulations", type=int, default=500, help="Simulaciones para bandas")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, output_dir = get_default_paths()

    artifacts = run_pipeline(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        horizon=args.horizon,
        no_forecast=args.no_forecast,
        n_simulations=args.n_simulations,
    )

    start, end = artifacts["range"]
    print(f"\nOK | meses={artifacts['months']} | rango={start.date()} → {end.date()} | outputs={output_dir}")


if __name__ == "__main__":
    main()
