"""
Motor de preguntas y respuestas (RAG).

Soporta 3 modos de LLM:
  - azure: Azure OpenAI Chat
  - local: Ollama local (sin API)
  - extractive: sin LLM; genera una respuesta extractiva basada en los campos
    recuperados (útil para pruebas/offline y para evitar fallos cuando faltan credenciales).

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict


# ---- Configuración LLM ----
LLM_MODE = os.getenv("LLM_MODE", "azure").lower()  # azure | local | extractive

# Azure OpenAI Chat
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ---- Prompt base ----
SYSTEM_PROMPT = """
Eres un asistente legal amigable. Tu trabajo es explicar casos jurídicos
en lenguaje sencillo y coloquial, como si le explicaras a un amigo que
no estudió derecho.

Reglas estrictas:
1. Responde ÚNICAMENTE con base en la información que se te proporciona en el contexto.
2. Si la información no está en el contexto, di exactamente: "No encontré información sobre eso en los casos disponibles."
3. NO inventes datos, fechas, nombres ni sentencias.
4. Usa un lenguaje simple, cercano y directo. Evita términos jurídicos complejos.
5. Al final de cada respuesta, incluye siempre: "📄 Fuente: [Providencia] ([Fecha])"
6. Si hay varios casos relevantes, menciona cada uno con su fuente.
""".strip()


def format_context(retrieved_docs: List[Dict]) -> str:
    """Formatea documentos recuperados como contexto para el LLM."""
    if not retrieved_docs:
        return "No se encontraron casos relevantes."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"--- Caso {i} ---\n"
            f"Providencia: {doc.get('providencia','')}\n"
            f"Fecha: {doc.get('fecha','')}\n"
            f"Tema: {(doc.get('tema') or 'No disponible')[:300]}\n"
            f"Síntesis: {(doc.get('sintesis') or 'No disponible')[:800]}\n"
            f"Sentencia: {(doc.get('resuelve') or 'No disponible')[:700]}\n"
        )

    return "\n".join(context_parts)


def _validate_azure_env() -> None:
    missing = []
    if not AZURE_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_CHAT_DEPLOYMENT:
        missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if missing:
        raise RuntimeError(
            "Faltan variables de entorno para Azure OpenAI: " + ", ".join(missing)
        )


def call_llm_azure(messages: List[Dict]) -> str:
    """Llama a Azure OpenAI Chat Completions."""
    _validate_azure_env()

    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=900,
    )

    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    # Debug útil (puedes dejarlo, no molesta)
    print(f"[AZURE] finish_reason={getattr(choice, 'finish_reason', None)} | chars={len(text)}")

    if not text:
        # Si Azure responde vacío, hacemos fallback seguro para no romper la entrega
        return ""
    return text

    return response.choices[0].message.content


def call_llm_local(messages: List[Dict]) -> str:
    """Llama a un LLM local via Ollama."""
    import requests

    prompt = f"Sistema: {messages[0]['content']}\n\n"
    for msg in messages[1:]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        prompt += f"{role}: {msg['content']}\n"

    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": os.getenv("OLLAMA_MODEL", "mistral"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=800, # Se aumenta tiempo debido a capacidades locales, ajustar segun el caso.
    )

    response.raise_for_status()
    return response.json()["response"]


def _plain_decision(resuelve: str) -> str:
    t = (resuelve or "").upper()
    if any(k in t for k in ["CONCEDE", "AMPARA", "TUTELA", "SE PROTEGE"]):
        return "En resumen: la Corte le dio la razón a la parte que demandó (concedió la tutela/amparo)."
    if any(k in t for k in ["NIEGA", "SE NIEGA", "NO CONCEDE"]):
        return "En resumen: la Corte no le dio la razón (negó la tutela)."
    if "IMPROCED" in t:
        return "En resumen: la Corte dijo que ese camino no aplicaba para este caso (lo declaró improcedente)."
    if any(k in t for k in ["ORDENA", "SE ORDENA", "DISPONE"]):
        return "En resumen: la Corte ordenó acciones concretas para corregir la situación."
    return "En resumen: la decisión está descrita en el apartado 'Sentencia/Resuelve'."


def answer_extractive(question: str, retrieved_docs: List[Dict]) -> str:
    """Respuesta sin LLM (fallback).

    La respuesta se arma con los campos recuperados (sintesis/resuelve) y se
    intenta mantener en lenguaje sencillo. Se adapta al tipo de pregunta:
      - si preguntan por "sentencia": prioriza resuelve/decisión
      - si preguntan por "de qué se trató" o "detalle": prioriza síntesis
    """
    if not retrieved_docs:
        return "No encontré información sobre eso en los casos disponibles."

    q = (question or "").lower()
    want_sentence = any(k in q for k in ["sentencia", "resuelve", "decisión", "decision"]) and "de qué" not in q and "detalle" not in q
    want_story = any(k in q for k in ["de qué", "de que", "trató", "trato", "qué pasó", "que paso", "detalle"])

    lines: list[str] = []
    for doc in retrieved_docs:
        providencia = doc.get("providencia") or "(sin providencia)"
        fecha = doc.get("fecha") or "(sin fecha)"

        sintesis = (doc.get("sintesis") or "").strip()
        resuelve = (doc.get("resuelve") or "").strip()

        # Recortes para que el output sea legible
        sintesis_short = (sintesis[:520] + "…") if len(sintesis) > 520 else sintesis
        resuelve_one_line = " ".join(resuelve.split())
        resuelve_short = (resuelve_one_line[:420] + "…") if len(resuelve_one_line) > 420 else resuelve_one_line

        decision = _plain_decision(resuelve) if resuelve else ""

        block = [f"- **{providencia} ({fecha})**"]

        if want_story and sintesis_short:
            block.append(f"  - **¿De qué se trató?:** {sintesis_short}")

        if (want_sentence or not want_story) and resuelve_short:
            block.append(f"  - **Sentencia (texto):** {resuelve_short}")
            if decision:
                block.append(f"  - {decision}")

        # Si la pregunta es de “detalle”, mostramos ambos (si existen)
        if ("detalle" in q or "qué pasó" in q or "que paso" in q) and sintesis_short and resuelve_short:
            # ya están incluidos por reglas anteriores, no hacemos nada extra
            pass

        block.append(f"  - 📄 Fuente: {providencia} ({fecha})")
        lines.append("\n".join(block))

    return "\n\n".join(lines)




def answer(question: str, retrieved_docs: List[Dict], verbose: bool = True) -> str:
    """Genera una respuesta usando el modo configurado."""
    context = format_context(retrieved_docs)

    if verbose:
        print(f"\n{'='*50}")
        print(f"PREGUNTA: {question}")
        print(f"DOCUMENTOS RECUPERADOS: {len(retrieved_docs)}")
        for doc in retrieved_docs:
            print(f"  - {doc.get('providencia','')} (score: {doc.get('score', 0):.3f})")
        print("=" * 50)

    if LLM_MODE == "extractive":
        return answer_extractive(question, retrieved_docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Aquí están los casos relevantes que encontré:\n\n"
                f"{context}\n\n"
                f"Pregunta: {question}"
            ),
        },
    ]

    if LLM_MODE == "azure":
        out = call_llm_azure(messages)
        if not (out or "").strip():
            return answer_extractive(question, retrieved_docs)
        return out

    if LLM_MODE == "local":
        try:
            out = call_llm_local(messages)
        except Exception as e:
            out = ""
        if not (out or "").strip():
            return answer_extractive(question, retrieved_docs)
        return out

    raise ValueError("LLM_MODE inválido. Usa: azure | local | extractive")


def save_answers(qa_pairs: List[Dict], output_path: str = "outputs/respuestas.md") -> None:
    """Guarda las respuestas en un archivo Markdown bien formateado."""
    Path(output_path).parent.mkdir(exist_ok=True)

    lines = [
        "# Respuestas - Asesor Legal IA\n",
        "**Prueba Técnica 2 – DataKnow SAS**\n",
        "---\n",
    ]

    for i, pair in enumerate(qa_pairs, 1):
        lines.append(f"## Pregunta {i}\n")
        lines.append(f"**{pair['question']}**\n")
        lines.append(f"\n{pair['answer']}\n")

        if pair.get("docs"):
            lines.append("\n**Casos consultados (retrieval):**\n")
            for doc in pair["docs"]:
                lines.append(
                    f"- {doc.get('providencia','')} ({doc.get('fecha','')}) – score: {doc.get('score', 0):.3f}\n"
                )

        lines.append("\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n✅ Respuestas guardadas en: {output_path}")
