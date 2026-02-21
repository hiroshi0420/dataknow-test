"""
qa.py
-----
Motor de preguntas y respuestas (RAG).
Recupera documentos relevantes y genera respuestas en lenguaje coloquial.

Soporta dos modos de LLM:
  - LOCAL: usa un modelo open source vía Ollama o HuggingFace
  - AZURE: usa Azure OpenAI Chat (bono +5%)
"""

import os
from pathlib import Path

# ---- Configuración LLM ----
LLM_MODE = os.getenv("LLM_MODE", "azure")  # "local" o "azure"

# Azure OpenAI Chat
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# ---- Prompt base ----
# Este prompt es clave para evitar alucinaciones y respuestas en jerga jurídica.
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
"""

def format_context(retrieved_docs: list[dict]) -> str:
    """
    Formatea los documentos recuperados como contexto para el LLM.
    
    Args:
        retrieved_docs: Lista de dicts de index.search()
        
    Returns:
        String con el contexto estructurado
    """
    if not retrieved_docs:
        return "No se encontraron casos relevantes."
    
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"--- Caso {i} ---\n"
            f"Providencia: {doc['providencia']}\n"
            f"Fecha: {doc['fecha']}\n"
            f"Tema: {doc['tema'][:300] if doc['tema'] else 'No disponible'}\n"
            f"Síntesis: {doc['sintesis'][:600] if doc['sintesis'] else 'No disponible'}\n"
            f"Sentencia: {doc['resuelve'][:400] if doc['resuelve'] else 'No disponible'}\n"
        )
    
    return "\n".join(context_parts)


def call_llm_azure(messages: list[dict]) -> str:
    """
    Llama a Azure OpenAI Chat Completions.
    
    Requiere variables de entorno:
      AZURE_OPENAI_ENDPOINT
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_CHAT_DEPLOYMENT
    """
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2024-02-01",
    )
    
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.1,  # Bajo para respuestas más precisas y menos creativas
        max_tokens=800,
    )
    
    return response.choices[0].message.content


def call_llm_local(messages: list[dict]) -> str:
    """
    Llama a un LLM local via Ollama.
    
    Instalar Ollama: https://ollama.ai
    Modelo recomendado: ollama pull mistral o ollama pull llama3
    
    También puedes adaptar esto para HuggingFace Inference API.
    """
    import requests
    
    # Convertir formato OpenAI a Ollama
    prompt = f"Sistema: {messages[0]['content']}\n\n"
    for msg in messages[1:]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        prompt += f"{role}: {msg['content']}\n"
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": os.getenv("OLLAMA_MODEL", "mistral"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=120,
    )
    
    return response.json()["response"]


def answer(
    question: str,
    retrieved_docs: list[dict],
    verbose: bool = True,
) -> str:
    """
    Genera una respuesta a la pregunta usando los documentos recuperados.
    
    Args:
        question: Pregunta del usuario
        retrieved_docs: Documentos relevantes de index.search()
        verbose: Si True, imprime el contexto usado
        
    Returns:
        Respuesta en lenguaje coloquial con fuentes
    """
    context = format_context(retrieved_docs)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"PREGUNTA: {question}")
        print(f"DOCUMENTOS RECUPERADOS: {len(retrieved_docs)}")
        for doc in retrieved_docs:
            print(f"  - {doc['providencia']} (score: {doc.get('score', 0):.3f})")
        print('='*50)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Aquí están los casos relevantes que encontré:\n\n"
                f"{context}\n\n"
                f"Pregunta: {question}"
            ),
        },
    ]
    
    if LLM_MODE == "azure":
        response = call_llm_azure(messages)
    else:
        response = call_llm_local(messages)
    
    return response


def save_answers(qa_pairs: list[dict], output_path: str = "outputs/respuestas.md"):
    """
    Guarda las respuestas en un archivo Markdown bien formateado.
    
    Args:
        qa_pairs: Lista de dicts con keys 'question', 'answer', 'docs'
        output_path: Ruta del archivo de salida
    """
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
            lines.append("\n**Casos consultados:**\n")
            for doc in pair["docs"]:
                lines.append(f"- {doc['providencia']} ({doc['fecha']}) – score: {doc.get('score', 0):.3f}\n")
        
        lines.append("\n---\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"\n✅ Respuestas guardadas en: {output_path}")