"""
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
OUTPUT_DIR = Path("outputs")

# Paths "base" (se recalculan por método dentro de build_index)
INDEX_PATH = OUTPUT_DIR / "faiss_index.pkl"
DOCS_PATH = OUTPUT_DIR / "documents.json"
TFIDF_VECTORIZER_PATH = OUTPUT_DIR / "tfidf_vectorizer.pkl"
META_PATH = OUTPUT_DIR / "index_meta.json"

# Local (si está disponible)
LOCAL_MODEL_NAME = os.getenv(
    "LOCAL_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Cache (en proceso)
_LOCAL_ST_MODEL = None
_TFIDF_VECTORIZER = None


def _safe_token(s: str) -> str:
    return (s or "").replace("/", "_").replace("\\", "_").replace(" ", "_").strip("_") or "default"


def _artifact_suffix(local_backend: str) -> str:
    """
    local_backend: "st" o "tfidf"
    """
    if EMBEDDING_MODE == "azure":
        return f"azure_{_safe_token(AZURE_EMBED_DEPLOYMENT)}"
    return f"local_{local_backend}"


def _paths_for(sfx: str) -> dict:
    return {
        "index": OUTPUT_DIR / f"faiss_index_{sfx}.pkl",
        "docs": OUTPUT_DIR / f"documents_{sfx}.json",
        "meta": OUTPUT_DIR / f"index_meta_{sfx}.json",
        "tfidf": OUTPUT_DIR / f"tfidf_vectorizer_{sfx}.pkl",
    }


def _purge(paths: dict) -> None:
    for key in ("index", "docs", "meta", "tfidf"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                Path(p).unlink()
            except Exception:
                # best effort
                pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


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

    # Si no estamos refitteando, exigimos que el vectorizer exista en disco.
    if TFIDF_VECTORIZER_PATH.exists() and not force_refit:
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            _TFIDF_VECTORIZER = pickle.load(f)
        return _TFIDF_VECTORIZER

    if not force_refit and not TFIDF_VECTORIZER_PATH.exists():
        raise RuntimeError(
            f"No existe TF-IDF vectorizer en {TFIDF_VECTORIZER_PATH}. "
            "Primero construya el índice (build_index) para que quede fitteado sobre el corpus."
        )

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
        # best-effort: cargar vectorizer ya fitteado en disco
        vectorizer = _fit_or_load_tfidf(texts, force_refit=False)
    else:
        vectorizer = _fit_or_load_tfidf(fit_texts_for_tfidf, force_refit=True)

    X = vectorizer.transform(texts)  # sparse
    # Para FAISS necesitamos dense float32
    return X.astype(np.float32).toarray()

def _chunk_text(text: str, max_chars: int = 3500, overlap: int = 200) -> List[str]:
    """
    Chunking simple por caracteres (proxy de tokens).
    - max_chars: tamaño máximo del chunk
    - overlap: solape para no cortar ideas a la mitad
    """
    t = (text or "").strip()
    if not t:
        return [""]

    if len(t) <= max_chars:
        return [t]

    chunks = []
    start = 0
    while start < len(t):
        end = min(start + max_chars, len(t))
        chunks.append(t[start:end])
        if end == len(t):
            break
        start = max(0, end - overlap)
    return chunks

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

    # 1) Expandimos documentos -> chunks
    chunk_texts: List[str] = []
    chunk_doc_ids: List[int] = []

    for doc_i, t in enumerate(texts):
        for ch in _chunk_text(t, max_chars=3500, overlap=200):
            chunk_texts.append(ch)
            chunk_doc_ids.append(doc_i)

    # 2) Embeddings por chunk (batch)
    all_chunk_embs: List[List[float]] = []
    batch_size = 16
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=AZURE_EMBED_DEPLOYMENT)
        all_chunk_embs.extend([item.embedding for item in response.data])
        print(f"Embeddings (chunks): {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)}")

    chunk_embs = np.array(all_chunk_embs, dtype=np.float32)

    # 3) Agregación chunk -> doc (promedio)
    dim = chunk_embs.shape[1]
    doc_sum = np.zeros((len(texts), dim), dtype=np.float32)
    doc_cnt = np.zeros((len(texts),), dtype=np.int32)

    for emb, doc_i in zip(chunk_embs, chunk_doc_ids):
        doc_sum[doc_i] += emb
        doc_cnt[doc_i] += 1

    # Evita división por cero
    doc_cnt = np.maximum(doc_cnt, 1).astype(np.float32)
    doc_embs = doc_sum / doc_cnt[:, None]

    return doc_embs.astype(np.float32)

def build_index(documents: List[Dict], force_rebuild: bool = False):
    """Construye o carga el índice FAISS."""
    import faiss

    global INDEX_PATH, DOCS_PATH, TFIDF_VECTORIZER_PATH, META_PATH, _TFIDF_VECTORIZER

    # Determina backend local real (solo se usa si EMBEDDING_MODE=local)
    if EMBEDDING_MODE == "azure":
        local_backend = "na"  # no cargues ST
    else:
        local_backend = "st" if _try_load_sentence_transformer() is not None else "tfidf"
    sfx = _artifact_suffix(local_backend)
    paths = _paths_for(sfx)

    # Reasigna paths globales para que el resto del módulo (TF-IDF) quede consistente
    INDEX_PATH = paths["index"]
    DOCS_PATH = paths["docs"]
    META_PATH = paths["meta"]
    TFIDF_VECTORIZER_PATH = paths["tfidf"]
    _TFIDF_VECTORIZER = None  # evita cache cruzada si cambiaste de backend/método

    if INDEX_PATH.exists() and DOCS_PATH.exists() and META_PATH.exists() and not force_rebuild:
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            meta_ok = (
                meta.get("embedding_mode") == EMBEDDING_MODE
                and int(meta.get("dim", -1)) > 0
                and meta.get("suffix") == sfx
            )
            if meta_ok:
                print(f"Cargando índice existente desde disco ({INDEX_PATH.name})...")
                with open(INDEX_PATH, "rb") as f:
                    index = pickle.load(f)
                with open(DOCS_PATH, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                print(f"Índice cargado: {index.ntotal} vectores (dim={meta.get('dim')}) [{sfx}]")
                return index, documents

            print("Índice existente no coincide con la configuración actual. Se regenerará.")
        except Exception as e:
            print(f"⚠️ Falló la carga del índice ({e}). Se regenerará.")
            _purge(paths)
            force_rebuild = True

    print(f"Construyendo índice FAISS en modo: {EMBEDDING_MODE} [{sfx}]")

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

    meta = {
        "suffix": sfx,
        "embedding_mode": EMBEDDING_MODE,
        "local_backend": local_backend if EMBEDDING_MODE != "azure" else None,
        "local_model_name": LOCAL_MODEL_NAME if EMBEDDING_MODE != "azure" else None,
        "azure_embed_deployment": AZURE_EMBED_DEPLOYMENT if EMBEDDING_MODE == "azure" else None,
        "dim": dim,
        "n_docs": len(documents),
    }

    # Escritura atómica (evita archivos corruptos si se interrumpe el proceso)
    _atomic_write_bytes(INDEX_PATH, pickle.dumps(index))
    _atomic_write_text(DOCS_PATH, json.dumps(documents, ensure_ascii=False, indent=2))
    _atomic_write_text(META_PATH, json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"Índice construido y guardado: {index.ntotal} vectores (dim={dim}) [{sfx}]")
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
