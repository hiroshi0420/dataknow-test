import pandas as pd
import pytest

from src.transform import compute_equipment_costs, describe_series


@pytest.fixture
def sample_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "X": [100.0, 110.0, 105.0, 95.0, 120.0, 115.0],
            "Y": [500.0, 510.0, 490.0, 505.0, 520.0, 515.0],
            "Z": [2000.0, 2100.0, 1950.0, 2050.0, 2200.0, 2150.0],
        },
        index=dates,
    )


def test_equipo1_formula(sample_df):
    df = compute_equipment_costs(sample_df)
    expected = 0.20 * sample_df["X"] + 0.80 * sample_df["Y"]
    pd.testing.assert_series_equal(df["equipo1"], expected, check_names=False)


def test_equipo2_formula(sample_df):
    df = compute_equipment_costs(sample_df)
    expected = (sample_df["X"] + sample_df["Y"] + sample_df["Z"]) / 3
    pd.testing.assert_series_equal(df["equipo2"], expected, check_names=False)


def test_describe_series_columns(sample_df):
    df = compute_equipment_costs(sample_df)
    stats = describe_series(df)
    for col in ["Media", "Desv. Estándar", "Mínimo", "Máximo", "Coef. Variación (%)"]:
        assert col in stats.columns
