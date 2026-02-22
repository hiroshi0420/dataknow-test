"""
Orquestador principal de la Prueba Técnica 2 DataKnow SAS.

Responde las 5 preguntas requeridas usando un pipeline RAG:
  1) Cargar y preparar documentos  (src/ingest.py)
  2) Embeddings + indexación        (src/index.py)
  3) Retrieval + respuesta          (src/qa.py)

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Asegura que src/ esté en el path
ROOT = Path(__file__).parent
# Carga variables desde .env si existe (para no exportar a mano)
load_dotenv(ROOT / ".env")
sys.path.append(str(ROOT / "src"))

from ingest import load_excel, build_documents, filter_by_keywords
from index import build_index, search
from qa import answer, save_answers

#  Keywords para filtros determinísticos, se pueden ampliar o modificar
REDES_KEYWORDS = [
    "facebook",
    "instagram",
    "twitter",
    "whatsapp",
    "tiktok",
    "youtube",
    "linkedin",
    "red social",
]
ACOSO_KEYWORDS = ["acoso escolar", "bullying", "ciberacoso", "matoneo", "hostigamiento escolar"]
PIAR_KEYWORDS = ["PIAR", "plan individualizado"]


def _pick_unique_docs(docs: list[dict], max_n: int) -> list[dict]:
    """Evita repetir providencias si el retrieval trae duplicados."""
    seen = set()
    out = []
    for d in docs:
        p = d.get("providencia")
        if p and p not in seen:
            seen.add(p)
            out.append(d)
        if len(out) >= max_n:
            break
    return out

def _pick_best_acoso(docs: list[dict]) -> list[dict]:
    """Para preguntas 3 y 4 normalmente quieren solo *un* caso.
    Elegimos el más claramente relacionado con acoso escolar (matching simple con palabras claves).
    """
    if not docs:
        return []

    priority_terms = [
        "acoso escolar",
        "ciberacoso",
        "bullying",
        "matoneo",
    ]

    def score(d: dict) -> int:
        hay = ((d.get("tema") or "") + " " + (d.get("sintesis") or "") + " " + (d.get("resuelve") or "")).lower()
        s = 0
        # Más peso a términos más específicos
        for weight, term in zip(range(len(priority_terms), 0, -1), priority_terms):
            if term in hay:
                s += weight
        # Extra: si el tema menciona explícitamente "acoso escolar"
        tema = (d.get("tema") or "").lower()
        if "acoso escolar" in tema:
            s += 10
        return s

    best = sorted(docs, key=score, reverse=True)[0]
    return [best]



def main() -> None:
    print("\n" + "=" * 60)
    print("  ASESOR LEGAL IA – DataKnow SAS")
    print("  Prueba Técnica 2")
    print("=" * 60 + "\n")

    data_path = Path(os.getenv("DATA_PATH", str(ROOT / "Datos" / "sentencias_pasadas.xlsx")))
    if not data_path.exists():
        raise FileNotFoundError(
            f"No encuentro el Excel en: {data_path}. "
            "Setea DATA_PATH o verifica la carpeta Datos/."
        )

    # ---- PASO 1: Cargar y preparar documentos ----
    print("📂 Cargando datos...")
    df = load_excel(str(data_path))
    all_docs = build_documents(df)
    print(f"✅ {len(all_docs)} documentos preparados")

    # ---- PASO 2: Construir índice (o cargar si ya existe) ----
    print("\n🔍 Construyendo índice de embeddings...")
    index, all_docs = build_index(all_docs)

    # ---- PASO 3: Pre-filtros por tema ----
    redes_docs = filter_by_keywords(all_docs, REDES_KEYWORDS)
    acoso_docs = filter_by_keywords(all_docs, ACOSO_KEYWORDS)
    piar_docs = filter_by_keywords(all_docs, PIAR_KEYWORDS)

    print("\n📊 Documentos por tema:")
    print(f"   - Redes sociales: {len(redes_docs)}")
    print(f"   - Acoso escolar:  {len(acoso_docs)}")
    print(f"   - PIAR:           {len(piar_docs)}")

    # ---- PASO 4: Selección de 3 casos de redes sociales ----
    top_redes = search(
        query="demandas por publicaciones en redes sociales que vulneran derechos (buen nombre, honra, intimidad)",
        index=index,
        documents=all_docs,
        top_k=6,
        candidate_docs=redes_docs,
    )
    top_redes = _pick_unique_docs(top_redes, 3)

    # ---- PASO 5: Responder las 5 preguntas requeridas ----
    qa_pairs: list[dict] = []

    # Pregunta 1 (redes sociales)
    print("\n\n📝 Respondiendo pregunta 1 (sentencias - redes sociales)...")
    q1 = "¿Cuáles fueron las sentencias (decisiones finales) de estas 3 demandas relacionadas con redes sociales?"
    a1 = answer(q1, top_redes)
    qa_pairs.append({"question": q1, "answer": a1, "docs": top_redes})

    # Pregunta 2 (redes sociales)
    print("\n\n📝 Respondiendo pregunta 2 (¿de qué se trataron? - redes sociales)...")
    q2 = "¿De qué se trató cada una de esas 3 demandas de redes sociales?"
    a2 = answer(q2, top_redes)
    qa_pairs.append({"question": q2, "answer": a2, "docs": top_redes})

# Pregunta 3 y 4 (acoso escolar)
    print("\n\n📝 Respondiendo pregunta 3 (sentencia acoso escolar)...")
    acoso_retrieved = search(
        query="acoso escolar bullying en colegio demanda tutela",
        index=index,
        documents=all_docs,
        top_k=5,
        candidate_docs=acoso_docs,
    )
    acoso_best = _pick_best_acoso(acoso_retrieved)

    q3 = "¿Cuál fue la sentencia del caso que habla de acoso escolar?"
    a3 = answer(q3, acoso_best)
    qa_pairs.append({"question": q3, "answer": a3, "docs": acoso_best})

    print("\n\n📝 Respondiendo pregunta 4 (detalle acoso escolar)...")
    q4 = "¿Cuál es el detalle completo de la demanda relacionada con acoso escolar? ¿Qué pasó, quién demandó y por qué?"
    a4 = answer(q4, acoso_best)
    qa_pairs.append({"question": q4, "answer": a4, "docs": acoso_best})

    # Pregunta 5 (PIAR)
    print("\n\n📝 Respondiendo pregunta 5 (casos PIAR)...")
    piar_retrieved = search(
        query="PIAR plan individualizado de ajustes razonables educación inclusiva", 
        index=index,
        documents=all_docs,
        top_k=5,
        candidate_docs=piar_docs if piar_docs else all_docs,
    )

    q5 = "¿Existen casos que hablan sobre el PIAR? ¿De qué trataron y cuáles fueron sus sentencias?"
    a5 = answer(q5, piar_retrieved)
    qa_pairs.append({"question": q5, "answer": a5, "docs": piar_retrieved})

    # ---- PASO 6: Guardar resultados ----
    save_answers(qa_pairs, output_path=str(ROOT / "outputs" / "respuestas.md"))

    print("\n✅ ¡Proceso completado!")
    print(f"📄 Resultados guardados en: {ROOT / 'outputs' / 'respuestas.md'}")


if __name__ == "__main__":
    main()
