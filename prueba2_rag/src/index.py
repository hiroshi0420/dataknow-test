"""
index.py

Embeddings + búsqueda semántica.

- Modo azure: Azure OpenAI Embeddings.
- Modo local: intenta sentence-transformers. Si no está disponible , hace fallback a TF-IDF (scikit-learn), suficiente para
  esta PoC (329 docs) y 100% offline.

El índice FAISS y los documentos se guardan en disco.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Configuración 
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local").lower()  # local | azure
INDEX_PATH = Path("outputs/faiss_index.pkl")
DOCS_PATH = Path("outputs/documents.json")
TFIDF_VECTORIZER_PATH = Path("outputs/tfidf_vectorizer.pkl")
META_PATH = Path("outputs/index_meta.json")

# Local (si está disponible)
LOCAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Cache
_LOCAL_ST_MODEL = None
_TFIDF_VECTORIZER = None


def _try_load_sentence_transformer():
    """Carga sentence-transformers si está disponible."""
    global _LOCAL_ST_MODEL
    if _LOCAL_ST_MODEL is not None:
        return _LOCAL_ST_MODEL

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Cargando sentence-transformers: {LOCAL_MODEL_NAME}")
        _LOCAL_ST_MODEL = SentenceTransformer(LOCAL_MODEL_NAME)
        return _LOCAL_ST_MODEL
    except Exception:
        return None


def _fit_or_load_tfidf(texts: List[str], force_refit: bool = False):
    """Fit TF-IDF una vez y lo guarda para reutilizar."""
    global _TFIDF_VECTORIZER

    if _TFIDF_VECTORIZER is not None and not force_refit:
        return _TFIDF_VECTORIZER

    if TFIDF_VECTORIZER_PATH.exists() and not force_refit:
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            _TFIDF_VECTORIZER = pickle.load(f)
        return _TFIDF_VECTORIZER

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Config simple: n-grams para capturar frases como "acoso escolar" o "buen nombre"
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50_000,
        stop_words=None,
    )
    vectorizer.fit(texts)

    TFIDF_VECTORIZER_PATH.parent.mkdir(exist_ok=True)
    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    _TFIDF_VECTORIZER = vectorizer
    return vectorizer


def _embeddings_local(texts: List[str], fit_texts_for_tfidf: Optional[List[str]] = None) -> np.ndarray:
    """Embeddings locales: sentence-transformers o fallback TF-IDF."""
    st_model = _try_load_sentence_transformer()
    if st_model is not None:
        emb = st_model.encode(texts, show_progress_bar=True, batch_size=32)
        return np.array(emb, dtype=np.float32)

    # Fallback TF-IDF: Si nos pasan fit_texts_for_tfidf, significa que estamos construyendo el índice
    # principal y debemos fittear el vectorizer sobre el corpus completo.
    if fit_texts_for_tfidf is None:
        # best-effort: cargar vectorizer ya fitteado
        vectorizer = _fit_or_load_tfidf(texts, force_refit=False)
    else:
        vectorizer = _fit_or_load_tfidf(fit_texts_for_tfidf, force_refit=True)

    X = vectorizer.transform(texts)  # sparse
    # Para FAISS necesitamos dense float32
    return X.astype(np.float32).toarray()


def _validate_azure_env() -> None:
    missing = []
    if not AZURE_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_EMBED_DEPLOYMENT:
        missing.append("AZURE_OPENAI_EMBED_DEPLOYMENT")
    if missing:
        raise RuntimeError(
            "Faltan variables de entorno para Azure OpenAI embeddings: " + ", ".join(missing)
        )


def _embeddings_azure(texts: List[str]) -> np.ndarray:
    _validate_azure_env()
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    embeddings: List[List[float]] = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=AZURE_EMBED_DEPLOYMENT)
        embeddings.extend([item.embedding for item in response.data])
        print(f"Embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(embeddings, dtype=np.float32)


def build_index(documents: List[Dict], force_rebuild: bool = False):
    """Construye o carga el índice FAISS."""
    import faiss

    if INDEX_PATH.exists() and DOCS_PATH.exists() and META_PATH.exists() and not force_rebuild:
        # Cargar meta y validar que coincida con el modo actual
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if meta.get("embedding_mode") == EMBEDDING_MODE and int(meta.get("dim", -1)) > 0:
            print("Cargando índice existente desde disco...")
            with open(INDEX_PATH, "rb") as f:
                index = pickle.load(f)
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)
            print(f"Índice cargado: {index.ntotal} vectores (dim={meta.get('dim')})")
            return index, documents
        else:
            print("Índice existente no coincide con el modo actual. Se regenerará.")

    print(f"Construyendo índice FAISS en modo: {EMBEDDING_MODE}")

    texts = [doc["text"] for doc in documents]

    if EMBEDDING_MODE == "azure":
        embeddings = _embeddings_azure(texts)
    else:
        embeddings = _embeddings_local(texts, fit_texts_for_tfidf=texts)

    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    META_PATH.write_text(json.dumps({"embedding_mode": EMBEDDING_MODE, "dim": dim}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Índice construido y guardado: {index.ntotal} vectores (dim={dim})")
    return index, documents


def _embed_query(query: str) -> np.ndarray:
    if EMBEDDING_MODE == "azure":
        q = _embeddings_azure([query])
    else:
        q = _embeddings_local([query])
    return np.array(q, dtype=np.float32)


def search(
    query: str,
    index,
    documents: List[Dict],
    top_k: int = 5,
    candidate_docs: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Búsqueda semántica por similitud coseno."""

    import faiss

    # Si hay pre-filtro, construimos índice temporal (pequeño) con embeddings consistentes.
    if candidate_docs is not None and len(candidate_docs) > 0:
        texts = [doc["text"] for doc in candidate_docs]
        if EMBEDDING_MODE == "azure":
            emb = _embeddings_azure(texts)
        else:
            # Importante: si estamos en TF-IDF fallback, reutiliza el vectorizer fitteado.
            emb = _embeddings_local(texts)
        emb = np.array(emb, dtype=np.float32)
        faiss.normalize_L2(emb)
        temp_index = faiss.IndexFlatIP(emb.shape[1])
        temp_index.add(emb)
        search_index = temp_index
        search_docs = candidate_docs
    else:
        search_index = index
        search_docs = documents

    q_emb = _embed_query(query)
    faiss.normalize_L2(q_emb)

    k = min(top_k, search_index.ntotal)
    scores, indices = search_index.search(q_emb, k)

    results: List[Dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            doc = dict(search_docs[idx])
            doc["score"] = float(score)
            results.append(doc)

    return results
