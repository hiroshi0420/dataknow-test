# Prueba Técnica 2 – Asesor Legal IA


Sistema RAG (Retrieval-Augmented Generation) para consultar histórico de sentencias legales y responder preguntas en lenguaje coloquial.

---

## Arquitectura

```
Excel (329 filas) 
    → Documentos de texto estructurado   [ingest.py]
    → Embeddings + Índice FAISS          [index.py]
    → Retrieval + Filtro por keywords    [index.py]
    → LLM (Azure OpenAI / Local)         [qa.py]
    → Respuestas coloquiales + Fuentes   [outputs/respuestas.md]
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Configuración

```bash
cp .env.example .env
# Por favor editar .env con tus credenciales
```

### Modo local (sin costo, sin API)
```env
EMBEDDING_MODE=local
LLM_MODE=local
```
Requiere [Ollama](https://ollama.ai) corriendo localmente con `ollama pull mistral`. (se usa mistral debido a su bajo consumo de recursos y buen rendimiento)

### Modo Azure
```env
EMBEDDING_MODE=azure
LLM_MODE=azure
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-ada-002
```

---

## Ejecución

```bash
# Correr el pipeline completo (responde las 5 preguntas)
python main.py

# Ver resultados
cat outputs/respuestas.md
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Preguntas:

1. ¿Cuáles son las sentencias de 3 demandas? (redes sociales)
2. ¿De qué se trataron esas 3 demandas?
3. ¿Cuál fue la sentencia del caso de acoso escolar?
4. ¿Cuál es el detalle de la demanda de acoso escolar?
5. ¿Existen casos sobre PIAR? ¿De qué trataron y cuáles fueron sus sentencias?

---

## Estructura del proyecto

```
prueba_2_rag/
├── main.py                  ← Orquestador principal
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py            ← Carga Excel y construye documentos
│   ├── index.py             ← Embeddings + FAISS
│   └── qa.py                ← RAG: retrieval + LLM + respuesta
├── tests/
│   └── test_ingest.py       ← Tests unitarios
└── outputs/
    └── respuestas.md        ← Respuestas generadas
```

---

## Supuestos

- El Excel `sentencias_pasadas.xlsx` contiene 329 casos con columnas: `Providencia`, `Fecha Sentencia`, `Tema - subtema`, `resuelve`, `sintesis`.
- Para las 3 demandas de redes sociales se eligieron los casos más semánticamente relevantes mediante retrieval automático sobre el subconjunto filtrado por keywords.
- Se confirmó la existencia de casos sobre acoso escolar (9 documentos) y PIAR (2 documentos) en el Excel original.
- Las respuestas del LLM están limitadas estrictamente al contenido recuperado para evitar alucinaciones.