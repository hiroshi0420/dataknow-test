"""
main.py
-------
Orquestador principal de la Prueba Técnica 2 – DataKnow SAS.

Responde las 5 preguntas requeridas usando el pipeline RAG completo:
  1. Cargar y preparar documentos  (ingest.py)
  2. Embedir e indexar              (index.py)
  3. Recuperar + responder          (qa.py)

Uso:
    python main.py

Variables de entorno necesarias para modo Azure:
    AZURE_OPENAI_ENDPOINT=https://...
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
    AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-ada-002
    EMBEDDING_MODE=azure
    LLM_MODE=azure
"""

import sys
from pathlib import Path

# Verificación de rutas
sys.path.append(str(Path(__file__).parent / "src"))

from ingest import load_excel, build_documents, filter_by_keywords
from index import build_index, search
from qa import answer, save_answers

# Ruta de los datos
DATA_PATH = "Datos/sentencias_pasadas.xlsx"  # Ajustar según tu entorno

# Keywords para filtros determinísticos
REDES_KEYWORDS = [
    "redes sociales", "facebook", "instagram", "twitter", "whatsapp",
    "tiktok", "youtube", "linkedin", "red social", "internet"
]
ACOSO_KEYWORDS = ["acoso escolar", "bullying", "ciberacoso", "matoneo"]
PIAR_KEYWORDS = ["PIAR", "plan individualizado", "ajustes razonables"]


def main():
    print("\n" + "="*60)
    print("  ASESOR LEGAL IA – DataKnow SAS")
    print("  Prueba Técnica 2")
    print("="*60 + "\n")
    
    # PASO 1: Cargar y preparar documentos
    print("📂 Cargando datos...")
    df = load_excel(DATA_PATH)
    all_docs = build_documents(df)
    print(f"✅ {len(all_docs)} documentos preparados")
    
    # PASO 2: Construir índice (o cargar si ya existe)
    print("\n🔍 Construyendo índice de embeddings...")
    index, all_docs = build_index(all_docs)
    
    # PASO 3: Pre-filtros por tema
    redes_docs = filter_by_keywords(all_docs, REDES_KEYWORDS)
    acoso_docs = filter_by_keywords(all_docs, ACOSO_KEYWORDS)
    piar_docs = filter_by_keywords(all_docs, PIAR_KEYWORDS)
    
    print(f"\n📊 Documentos por tema:")
    print(f"   - Redes sociales: {len(redes_docs)}")
    print(f"   - Acoso escolar:  {len(acoso_docs)}")
    print(f"   - PIAR:           {len(piar_docs)}")
    
    # PASO 4: Selección de 3 casos de redes sociales
    # Elegimos los 3 más relevantes semánticamente para la demo
    # NOTA: Puedes elegir manualmente editando esta sección
    top_redes = search(
        query="demandas por publicaciones en redes sociales que vulneran derechos",
        index=index,
        documents=all_docs,
        top_k=3,
        candidate_docs=redes_docs,
    )
    
    #PASO 5: Responder las 5 preguntas requeridas
    qa_pairs = []
    
    # Pregunta 1 y 2 van juntas: sentencias + de qué se trataron (3 casos redes sociales)
    print("\n\n📝 Respondiendo pregunta 1 y 2 (3 demandas de redes sociales)...")
    q1 = "¿Cuáles son las sentencias de estas 3 demandas y de qué se trataron?"
    a1 = answer(q1, top_redes)
    qa_pairs.append({
        "question": "¿Cuáles son las sentencias de 3 demandas? (elegidas del tema redes sociales) / ¿De qué se trataron?",
        "answer": a1,
        "docs": top_redes,
    })
    print(f"\n💬 Respuesta:\n{a1}")
    
    # Pregunta 3: Sentencia del caso de acoso escolar
    print("\n\n📝 Respondiendo pregunta 3 (sentencia acoso escolar)...")
    acoso_retrieved = search(
        query="acoso escolar bullying en colegio demanda tutela",
        index=index,
        documents=all_docs,
        top_k=3,
        candidate_docs=acoso_docs,
    )
    q3 = "¿Cuál fue la sentencia del caso que habla de acoso escolar?"
    a3 = answer(q3, acoso_retrieved)
    qa_pairs.append({
        "question": q3,
        "answer": a3,
        "docs": acoso_retrieved,
    })
    print(f"\n💬 Respuesta:\n{a3}")
    
    # Pregunta 4: Detalle del caso de acoso escolar
    print("\n\n📝 Respondiendo pregunta 4 (detalle acoso escolar)...")
    q4 = "¿Cuál es el detalle completo de la demanda relacionada con acoso escolar? ¿Qué pasó, quién demandó y por qué?"
    a4 = answer(q4, acoso_retrieved)
    qa_pairs.append({
        "question": q4,
        "answer": a4,
        "docs": acoso_retrieved,
    })
    print(f"\n💬 Respuesta:\n{a4}")
    
    # Pregunta 5: Casos sobre PIAR
    print("\n\n📝 Respondiendo pregunta 5 (casos PIAR)...")
    piar_retrieved = search(
        query="PIAR plan individualizado de ajustes razonables educación especial",
        index=index,
        documents=all_docs,
        top_k=5,
        candidate_docs=piar_docs if piar_docs else all_docs,
    )
    q5 = "¿Existen casos que hablan sobre el PIAR? ¿De qué trataron y cuáles fueron sus sentencias?"
    a5 = answer(q5, piar_retrieved)
    qa_pairs.append({
        "question": q5,
        "answer": a5,
        "docs": piar_retrieved,
    })
    print(f"\n💬 Respuesta:\n{a5}")
    
    # PASO 6: Guardar resultados
    save_answers(qa_pairs, output_path="outputs/respuestas.md")
    
    print("\n\n✅ ¡Proceso completado!")
    print("📄 Resultados guardados en: outputs/respuestas.md")


if __name__ == "__main__":
    main()