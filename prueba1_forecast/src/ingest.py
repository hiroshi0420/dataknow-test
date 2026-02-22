"""Ingestión y normalización de las series X, Y y Z.

Los tres archivos vienen con formatos distintos:
- X.csv: separador ',', fechas ISO, decimal '.'
- Y.csv: separador ';', fechas day-first, decimal ','
- Z.csv: columnas invertidas (Price,Date) con fechas ISO

Salida: DataFrame mensual (month-end) usando promedio mensual, alineado por
intersección temporal (solo meses donde existen X, Y y Z).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """Lee un CSV probando encodings comunes."""
    encodings = [kwargs.pop("encoding", None), "utf-8-sig", "utf-8", "latin-1"]
    last_err: Exception | None = None
    for enc in [e for e in encodings if e]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"No se pudo leer {path.name}. Último error: {last_err}")


def load_x(path: Path) -> pd.Series:
    df = _read_csv_with_fallback(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date").set_index("Date")
    s = pd.to_numeric(df["Price"], errors="coerce").rename("X")
    _validate(s, "X")
    return s


def load_y(path: Path) -> pd.Series:
    df = _read_csv_with_fallback(
        path,
        sep=";",
        decimal=",",
    )
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    df = df.sort_values("Date").set_index("Date")
    s = pd.to_numeric(df["Price"], errors="coerce").rename("Y")
    _validate(s, "Y")
    return s


def load_z(path: Path) -> pd.Series:
    df = _read_csv_with_fallback(path)

    # Normaliza headers típicos: Price,Date o Date,Price
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    if "Date" not in df.columns and "date" in [c.lower() for c in df.columns]:
        for c in df.columns:
            if c.lower() == "date":
                df = df.rename(columns={c: "Date"})
    if "Price" not in df.columns and "price" in [c.lower() for c in df.columns]:
        for c in df.columns:
            if c.lower() == "price":
                df = df.rename(columns={c: "Price"})

    # Si el archivo viene invertido (Price,Date) igual funciona por nombre
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"Z.csv debe tener columnas Date y Price. Encontré: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date").set_index("Date")
    s = pd.to_numeric(df["Price"], errors="coerce").rename("Z")
    _validate(s, "Z")
    return s


def to_monthly_mean(series: pd.Series) -> pd.Series:
    """Convierte una serie diaria a frecuencia mensual usando promedio mensual (month-end)."""
    return series.resample("ME").mean()


def load_all(data_dir: Path) -> pd.DataFrame:
    """Carga X/Y/Z, convierte a mensual y alinea por intersección temporal."""
    x = load_x(data_dir / "X.csv")
    y = load_y(data_dir / "Y.csv")
    z = load_z(data_dir / "Z.csv")

    x_m = to_monthly_mean(x)
    y_m = to_monthly_mean(y)
    z_m = to_monthly_mean(z)

    df = pd.concat([x_m, y_m, z_m], axis=1, join="inner")
    df.index.name = "month"
    return df


def _validate(series: pd.Series, name: str) -> None:
    if series.empty:
        raise ValueError(f"{name}: serie vacía")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name}: índice de fechas no está ordenado")
    if series.isna().any():
        # Se permite NA en origen; se maneja con inner join mensual.
        pass

