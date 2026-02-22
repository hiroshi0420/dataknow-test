import pytest
import pandas as pd
from pathlib import Path
from src.ingest import load_x, load_y, load_z, load_all

@pytest.fixture
def dummy_data_dir(tmp_path: Path):
    # Mock X.csv: fecha ISO, decimal '.', separador ','
    x_file = tmp_path / "X.csv"
    x_file.write_text("Date,Price\n2020-01-01,10.5\n2020-02-01,11.0\n", encoding="utf-8")

    # Mock Y.csv: fecha dayfirst, decimal ',', separador ';', codificacion latin-1
    y_file = tmp_path / "Y.csv"
    y_file.write_text("Date;Price\n31/01/2020;20,5\n29/02/2020;21,0\n", encoding="latin-1")

    # Mock Z.csv: columnas invertidas (Price, date), espacios en el encabezado, codificacion UTF-8-SIG
    z_file = tmp_path / "Z.csv"
    z_file.write_bytes(b"\xef\xbb\xbf Price , date \n 30.5 , 2020-01-01 \n 31.0 , 2020-02-01 \n")

    return tmp_path

def test_load_x(dummy_data_dir):
    series = load_x(dummy_data_dir / "X.csv")
    assert series.name == "X"
    assert len(series) == 2
    assert series.iloc[0] == 10.5
    assert series.index[0] == pd.Timestamp("2020-01-01")

def test_load_y(dummy_data_dir):
    series = load_y(dummy_data_dir / "Y.csv")
    assert series.name == "Y"
    assert len(series) == 2
    assert series.iloc[0] == 20.5
    assert series.index[0] == pd.Timestamp("2020-01-31")

def test_load_z(dummy_data_dir):
    series = load_z(dummy_data_dir / "Z.csv")
    assert series.name == "Z"
    assert len(series) == 2
    assert series.iloc[0] == 30.5
    assert series.index[0] == pd.Timestamp("2020-01-01")

def test_load_all(dummy_data_dir):
    # la carga deberia convertir los datos diarios a mensuales y unirlos
    df = load_all(dummy_data_dir)
    assert "X" in df.columns
    assert "Y" in df.columns
    assert "Z" in df.columns
    # Ambos meses deberian estar presentes porque se mapean a los ultimos dias de Enero y Febrero
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2020-01-31")
    assert df.index[1] == pd.Timestamp("2020-02-29")
    
    # Valores para Enero
    assert df.loc["2020-01-31", "X"] == 10.5
    assert df.loc["2020-01-31", "Y"] == 20.5
    assert df.loc["2020-01-31", "Z"] == 30.5

def test_load_z_missing_columns(tmp_path):
    z_file = tmp_path / "Z_bad.csv"
    z_file.write_text("Valor,Fecha\n10,2020-01-01\n")
    with pytest.raises(ValueError, match="Z.csv debe tener columnas Date y Price"):
         load_z(z_file)

def test_missing_file_raises_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="No se pudo leer missing.csv"):
         load_x(tmp_path / "missing.csv")
