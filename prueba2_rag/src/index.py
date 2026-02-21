"""
index.py
--------
Genera embeddings de los documentos y los almacena en un índice FAISS.
Soporta dos modos:
  - LOCAL: usa sentence-transformers (open source, sin costo)
  - AZURE: usa Azure OpenAI Embeddings (para el bono +5%)

El índice se guarda en disco para no tener que regenerarlo cada vez.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path

# ---- Configuración ----
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")  # "local" o "azure"
INDEX_PATH = Path("outputs/faiss_index.pkl")
DOCS_PATH = Path("outputs/documents.json")

# Modelo local recomendado (multilingüe, funciona bien en español)
LOCAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Azure OpenAI (solo si EMBEDDING_MODE="azure")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")


def get_embeddings_local(texts: list[str]) -> np.ndarray:
    """
    Genera embeddings usando sentence-transformers (sin costo, funciona offline).
    
    Instalar: pip install sentence-transformers
    """
    from sentence_transformers import SentenceTransformer
    
    print(f"Cargando modelo local: {LOCAL_MODEL_NAME}")
    model = SentenceTransformer(LOCAL_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings  # shape: (n_docs, embedding_dim)


def get_embeddings_azure(texts: list[str]) -> np.ndarray:
    """
    Genera embeddings usando Azure OpenAI (bono +5%).
    
    Requiere variables de entorno:
      AZURE_OPENAI_ENDPOINT
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_EMBED_DEPLOYMENT
      
    Instalar: pip install openai
    """
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2024-02-01",
    )
    
    embeddings = []
    batch_size = 16  # Azure tiene límite de tokens por request
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=AZURE_EMBED_DEPLOYMENT,
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")
    
    return np.array(embeddings, dtype=np.float32)


def build_index(documents: list[dict], force_rebuild: bool = False) -> tuple:
    """
    Construye el índice FAISS con los embeddings de los documentos.
    
    Si el índice ya existe en disco, lo carga directamente (más rápido).
    
    Args:
        documents: Lista de dicts de ingest.build_documents()
        force_rebuild: Si True, regenera el índice aunque exista en disco
        
    Returns:
        Tuple (index, documents) listos para búsqueda
    """
    import faiss
    
    # Si el índice ya existe, cargarlo
    if INDEX_PATH.exists() and DOCS_PATH.exists() and not force_rebuild:
        print("Cargando índice existente desde disco...")
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"Índice cargado: {index.ntotal} vectores")
        return index, documents
    
    print(f"Construyendo índice FAISS en modo: {EMBEDDING_MODE}")
    
    texts = [doc["text"] for doc in documents]
    
    # Generar embeddings según el modo configurado
    if EMBEDDING_MODE == "azure":
        embeddings = get_embeddings_azure(texts)
    else:
        embeddings = get_embeddings_local(texts)
    
    embeddings = embeddings.astype(np.float32)
    
    # Normalizar para similitud coseno (equivale a inner product con vectores unitarios)
    faiss.normalize_L2(embeddings)
    
    # Crear índice FAISS (Inner Product = cosine similarity con vectores normalizados)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"Índice construido: {index.ntotal} vectores de dimensión {dim}")
    
    # Guardar en disco
    INDEX_PATH.parent.mkdir(exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Índice guardado en: {INDEX_PATH}")
    return index, documents


def search(
    query: str,
    index,
    documents: list[dict],
    top_k: int = 5,
    candidate_docs: list[dict] = None,
) -> list[dict]:
    """
    Busca los documentos más similares a la query.
    
    Args:
        query: Pregunta en lenguaje natural
        index: Índice FAISS
        documents: Lista completa de documentos (para recuperar por ID)
        top_k: Número de resultados a retornar
        candidate_docs: Si se provee, busca solo en este subconjunto (pre-filtrado)
        
    Returns:
        Lista de documentos más relevantes con su score
    """
    import faiss
    
    # Si hay pre-filtro, crear un índice temporal solo con esos docs
    if candidate_docs is not None and len(candidate_docs) > 0:
        texts = [doc["text"] for doc in candidate_docs]
        
        if EMBEDDING_MODE == "azure":
            embeddings = get_embeddings_azure(texts)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(LOCAL_MODEL_NAME)
            embeddings = model.encode(texts)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(dim)
        temp_index.add(embeddings)
        search_index = temp_index
        search_docs = candidate_docs
    else:
        search_index = index
        search_docs = documents
    
    # Embedir la query
    if EMBEDDING_MODE == "azure":
        q_emb = get_embeddings_azure([query])
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(LOCAL_MODEL_NAME)
        q_emb = model.encode([query])
    
    q_emb = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    
    # Buscar
    k = min(top_k, search_index.ntotal)
    scores, indices = search_index.search(q_emb, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            doc = search_docs[idx].copy()
            doc["score"] = float(score)
            results.append(doc)
    
    return results


# ---------------------------------------------------------------------------
# Para prueba rápida
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from ingest import load_excel, build_documents, filter_by_keywords
    
    df = load_excel("../Datos/sentencias_pasadas.xlsx")
    docs = build_documents(df)
    
    index, docs = build_index(docs)
    
    # Prueba de búsqueda
    results = search("acoso escolar en colegio", index, docs, top_k=3)
    for r in results:
        print(f"\n[Score: {r['score']:.3f}] {r['providencia']} ({r['fecha']})")
        print(r['tema'][:200])