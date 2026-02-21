"""
tests/test_ingest.py
--------------------
Tests básicos para el módulo de ingesta.
Bono +5% calidad de código.
"""

import sys
import pandas as pd
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from ingest import load_excel, build_documents, filter_by_keywords

# Ruta al Excel de prueba (ajusta si es necesario)
EXCEL_PATH = "Datos/sentencias_pasadas.xlsx"


@pytest.fixture
def sample_df():
    """Crea un DataFrame de prueba pequeño."""
    return pd.DataFrame({
        "#": [1, 2, 3],
        "Relevancia": [1.0, 2.0, 3.0],
        "Providencia": ["T-001/22", "T-002/23", "T-003/24"],
        "Tipo": ["Tutela", "Tutela", "Tutela"],
        "Fecha Sentencia": ["2022-01-01", "2023-05-15", "2024-06-26"],
        "Tema - subtema": [
            "ACOSO ESCOLAR EN REDES SOCIALES",
            "PIAR EDUCACION ESPECIAL",
            "DERECHO AL BUEN NOMBRE",
        ],
        "resuelve": [
            "Se CONCEDE el amparo solicitado.",
            "Se NIEGA la acción de tutela.",
            "Se DECLARA IMPROCEDENTE.",
        ],
        "sintesis": [
            "Caso sobre acoso escolar y bullying en redes sociales.",
            "Caso sobre PIAR y plan de ajustes razonables.",
            "Caso sobre honor y redes sociales.",
        ],
    })


def test_build_documents_count(sample_df):
    """Debe construir un documento por fila válida."""
    docs = build_documents(sample_df)
    assert len(docs) == 3


def test_build_documents_has_required_fields(sample_df):
    """Cada documento debe tener los campos clave."""
    docs = build_documents(sample_df)
    for doc in docs:
        assert "id" in doc
        assert "providencia" in doc
        assert "text" in doc
        assert len(doc["text"]) > 10


def test_build_documents_text_contains_providencia(sample_df):
    """El texto del documento debe incluir la providencia."""
    docs = build_documents(sample_df)
    assert "T-001/22" in docs[0]["text"]


def test_filter_by_keywords_acoso(sample_df):
    """El filtro debe encontrar el documento sobre acoso escolar."""
    docs = build_documents(sample_df)
    filtered = filter_by_keywords(docs, ["acoso escolar", "bullying"])
    assert len(filtered) == 1
    assert filtered[0]["providencia"] == "T-001/22"


def test_filter_by_keywords_piar(sample_df):
    """El filtro debe encontrar el documento sobre PIAR."""
    docs = build_documents(sample_df)
    filtered = filter_by_keywords(docs, ["PIAR"])
    assert len(filtered) == 1
    assert filtered[0]["providencia"] == "T-002/23"


def test_filter_by_keywords_no_match(sample_df):
    """El filtro debe retornar lista vacía si no hay coincidencias."""
    docs = build_documents(sample_df)
    filtered = filter_by_keywords(docs, ["xyz_keyword_inexistente"])
    assert len(filtered) == 0


def test_filter_case_insensitive(sample_df):
    """El filtro debe ser case-insensitive."""
    docs = build_documents(sample_df)
    filtered = filter_by_keywords(docs, ["piar"])  # minúscula
    assert len(filtered) == 1