"""
ingest.py
---------
Carga el Excel de sentencias y construye una lista de documentos de texto
listos para ser embedidos e indexados.

Cada fila del Excel se convierte en un string estructurado que el LLM
puede leer fácilmente.

Nota:
- build_documents() soporta DataFrames ya normalizados (load_excel) y
  DataFrames crudos (útil para tests unitarios).
"""

from __future__ import annotations

import pandas as pd


def load_excel(path: str) -> pd.DataFrame:
    """Carga el Excel de sentencias y hace limpieza básica."""
    df = pd.read_excel(path)

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Convertir fecha a string legible
    if "fecha_sentencia" in df.columns:
        df["fecha_sentencia"] = pd.to_datetime(df["fecha_sentencia"], errors="coerce")
        df["fecha_str"] = df["fecha_sentencia"].dt.strftime("%d/%m/%Y")

    # Rellenar nulos con cadena vacía para evitar errores al armar el documento.
    text_cols = ["tema___subtema", "resuelve", "sintesis", "tipo", "providencia"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """Convierte cada fila del DataFrame en un dict listo para RAG."""

    def _get(row: pd.Series, *keys: str, default: str = "") -> str:
        """Obtiene el primer valor no vacío entre varias llaves."""
        for k in keys:
            if k in row and pd.notna(row[k]):
                v = str(row[k]).strip()
                if v and v.lower() != "nan":
                    return v
        return default

    documents: list[dict] = []

    for idx, row in df.iterrows():
        parts: list[str] = []

        # Soportar DF normalizado o crudo
        providencia = _get(row, "providencia", "Providencia")

        fecha_str = _get(row, "fecha_str")
        if not fecha_str:
            raw_fecha = _get(row, "fecha_sentencia", "Fecha Sentencia")
            if raw_fecha:
                fecha_dt = pd.to_datetime(raw_fecha, errors="coerce")
                if pd.notna(fecha_dt):
                    fecha_str = fecha_dt.strftime("%d/%m/%Y")

        tipo = _get(row, "tipo", "Tipo")
        tema = _get(row, "tema___subtema", "tema_-_subtema", "tema_subtema", "Tema - subtema")
        sintesis = _get(row, "sintesis", "Sintesis", "síntesis", "Síntesis")
        resuelve = _get(row, "resuelve", "Resuelve")

        if providencia:
            parts.append(f"Providencia: {providencia}")
        if fecha_str:
            parts.append(f"Fecha: {fecha_str}")
        if tipo:
            parts.append(f"Tipo: {tipo}")
        if tema:
            parts.append(f"Tema: {tema}")
        if sintesis:
            parts.append(f"Síntesis: {sintesis}")
        if resuelve:
            parts.append(f"Sentencia / Resuelve: {resuelve}")

        text = "\n".join(parts).strip()

        # Solo incluir filas con texto útil, (aqui se elimina toda la columan tipo)
        if len(text) < 50:
            continue

        documents.append(
            {
                "id": int(idx) if str(idx).isdigit() else idx,
                "providencia": providencia,
                "fecha": fecha_str,
                "tema": tema,
                "sintesis": sintesis,
                "resuelve": resuelve,
                "text": text,
            }
        )

    return documents


def filter_by_keywords(documents: list[dict], keywords: list[str]) -> list[dict]:
    """Filtro determinístico por palabras clave ANTES del retrieval semántico."""
    keywords_lower = [k.lower() for k in keywords]
    filtered: list[dict] = []

    for doc in documents:
        full_text = (doc.get("text") or "").lower()
        if any(kw in full_text for kw in keywords_lower):
            filtered.append(doc)

    return filtered

# Este bloque  lo use para pruebas locales, en productivo no es necesario.
if __name__ == "__main__":
    # Prueba rápida manual
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "Datos" / "sentencias_pasadas.xlsx"
    df_ = load_excel(str(data_path))
    docs_ = build_documents(df_)
    print(f"Filas cargadas: {len(df_)}")
    print(f"Documentos construidos: {len(docs_)}")

    redes_docs = filter_by_keywords(docs_, ["redes sociales", "facebook", "instagram", "twitter", "whatsapp", "tiktok"])
    print(f"Docs redes sociales: {len(redes_docs)}")
