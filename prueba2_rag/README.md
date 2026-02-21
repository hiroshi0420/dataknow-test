# Prueba Técnica 2 – Asesor Legal IA (RAG)
**DataKnow SAS | Científico de Datos**

PoC de RAG (Retrieval-Augmented Generation) para consultar un Excel de sentencias y responder preguntas en español coloquial, con fuentes.

## Arquitectura

```
Excel (329 filas)
  → Documentos de texto estructurado   [src/ingest.py]
  → Embeddings + Índice FAISS          [src/index.py]
  → Retrieval + filtro por keywords    [src/ingest.py + src/index.py]
  → LLM (Azure OpenAI / Ollama / fallback) [src/qa.py]
  → outputs/respuestas.md
```

## Requisitos
- Python 3.10+ (probado con 3.11)
- (Opcional) Docker / Docker Compose
- (Opcional) Azure OpenAI

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

```bash
cp .env.example .env
# edita .env con las variables de entorno locales o azure
```

Variables importantes:
- `DATA_PATH` (si el Excel no está en `Datos/`)
- `EMBEDDING_MODE`: `local` | `azure`
- `LLM_MODE`: `azure` | `local` | `extractive`

### Modo Azure

En `.env`:
Referenciar el ejemplo de .env.example y cambiar las variables de entorno por las de azure

### Modo local sin costo
- **Embeddings**: local (`sentence-transformers` si está disponible; si no, fallback a **TF-IDF** offline, esto se hace para evitar problemas con el aplicativo debido a que no es un documento con alto volumen de datos).
- **LLM**: 
  - `LLM_MODE=local` usa **Ollama** (http://localhost:11434)
  - `LLM_MODE=extractive` NO usa LLM (sirve para probar pipeline sin credenciales)

## Ejecución

```bash
python main.py
# outputs/respuestas.md
```

## Tests

```bash
pytest -v
```

## Docker (local)

1) Crear `.env`:
```bash
cp .env.example .env
```

2) Correr:
```bash
docker compose up --build
```

### Docker + Ollama (LLM local)

```bash
# En .env -> LLM_MODE=local
# y opcional: OLLAMA_MODEL=mistral

docker compose --profile ollama up --build
```

## Azure (idea de despliegue, opcional)

- Empaqueta la solución con Docker (realizado).
- Sube la imagen a Azure Container Registry (ACR).
- Despliega en Azure Container Apps con variables de entorno (Azure OpenAI).
