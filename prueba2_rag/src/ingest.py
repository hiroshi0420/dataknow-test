"""
ingest.py
---------
Carga el Excel de sentencias y construye una lista de documentos de texto
listos para ser embedidos e indexados.

Cada fila del Excel se convierte en un string estructurado que el LLM
puede leer fácilmente.
"""

import pandas as pd
from pathlib import Path


def load_excel(path: str) -> pd.DataFrame:
    """
    Carga el Excel de sentencias y hace limpieza básica.
    
    Args:
        path: Ruta al archivo sentencias_pasadas.xlsx
        
    Returns:
        DataFrame limpio con las columnas normalizadas
    """
    df = pd.read_excel(path)
    
    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    
    # Convertir fecha a string legible
    if "fecha_sentencia" in df.columns:
        df["fecha_sentencia"] = pd.to_datetime(df["fecha_sentencia"], errors="coerce")
        df["fecha_str"] = df["fecha_sentencia"].dt.strftime("%d/%m/%Y")
    
    # Rellenar nulos con cadena vacía para evitar errores al armar el documento
    text_cols = ["tema___subtema", "resuelve", "sintesis", "tipo", "providencia"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    
    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convierte cada fila del DataFrame en un diccionario con:
      - 'id': índice de la fila
      - 'providencia': código de la sentencia
      - 'fecha': fecha de la sentencia
      - 'text': texto completo para embedir
      
    El campo 'text' es el que se pasa al modelo de embeddings.
    El 'text' usa un formato estructurado para que el retrieval sea más preciso.
    
    Args:
        df: DataFrame limpio de load_excel()
        
    Returns:
        Lista de dicts, uno por sentencia
    """
    documents = []
    
    for idx, row in df.iterrows():
        
        # Construir el texto del documento con todos los campos relevantes
        # Esto es lo que el modelo de embeddings va a "leer" para decidir similitud
        parts = []
        
        if row.get("providencia"):
            parts.append(f"Providencia: {row['providencia']}")
        
        if row.get("fecha_str"):
            parts.append(f"Fecha: {row['fecha_str']}")
        
        if row.get("tipo"):
            parts.append(f"Tipo: {row['tipo']}")
        
        if row.get("tema___subtema"):
            parts.append(f"Tema: {row['tema___subtema']}")
        
        if row.get("sintesis"):
            parts.append(f"Síntesis: {row['sintesis']}")
        
        if row.get("resuelve"):
            parts.append(f"Sentencia / Resuelve: {row['resuelve']}")
        
        text = "\n".join(parts)
        
        # Solo incluir filas con texto útil
        if len(text.strip()) < 50:
            continue
        
        documents.append({
            "id": idx,
            "providencia": row.get("providencia", ""),
            "fecha": row.get("fecha_str", ""),
            "tema": row.get("tema___subtema", ""),
            "sintesis": row.get("sintesis", ""),
            "resuelve": row.get("resuelve", ""),
            "text": text,
        })
    
    return documents


def filter_by_keywords(documents: list[dict], keywords: list[str]) -> list[dict]:
    """
    Filtro determinístico por palabras clave ANTES del retrieval semántico.
    
    Esto es importante para preguntas sobre temas específicos (redes sociales,
    acoso escolar, PIAR) para asegurar que solo candidatos relevantes entren
    al LLM. Reduce alucinaciones.
    
    Args:
        documents: Lista completa de documentos
        keywords: Lista de palabras a buscar (case-insensitive)
        
    Returns:
        Subconjunto de documentos que contienen al menos una keyword
    """
    keywords_lower = [k.lower() for k in keywords]
    filtered = []
    
    for doc in documents:
        full_text = doc["text"].lower()
        if any(kw in full_text for kw in keywords_lower):
            filtered.append(doc)
    
    return filtered



if __name__ == "__main__":
    DATA_PATH = "../Datos/sentencias_pasadas.xlsx" 
    
    df = load_excel(DATA_PATH)
    print(f"Filas cargadas: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    
    docs = build_documents(df)
    print(f"Documentos construidos: {len(docs)}")
    print("\nEjemplo de documento:")
    print(docs[0]["text"][:500])
    
    # Probar filtro
    redes_docs = filter_by_keywords(docs, ["redes sociales", "facebook", "instagram", "twitter", "whatsapp", "tiktok"])
    print(f"\nDocumentos sobre redes sociales: {len(redes_docs)}")
    
    acoso_docs = filter_by_keywords(docs, ["acoso escolar", "bullying", "ciberacoso"])
    print(f"Documentos sobre acoso escolar: {len(acoso_docs)}")
    
    piar_docs = filter_by_keywords(docs, ["PIAR", "plan individualizado"])
    print(f"Documentos sobre PIAR: {len(piar_docs)}")