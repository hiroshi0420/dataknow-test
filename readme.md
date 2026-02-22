# DataKnow Tech Test – Workspace

Este repositorio contiene los entregables de la **Prueba Técnica DataKnow**.
El workspace está dividido en **dos proyectos independientes** orientados a demostrar diferentes capacidades analíticas (modelado clásico de series temporales y NLP avanzado con modelos generativos).

**Importante:** Para la ejecución mediante Docker, es necesario **ubicarse dentro de la carpeta del proyecto** correspondiente. No se debe ejecutar Docker Compose desde esta raíz.

---

## 📂 Estructura del Workspace

### 1️⃣ `prueba1_forecast/`
**Estimación de Costos de Equipos (Series de Tiempo)**
Pipeline batch reproducible diseñado para:
- Ingestión, limpieza y consolidación de X, Y, Z (solucionando problemas de delimitadores, encodings y columnas invertidas).
- Creación de resúmenes estadísticos e imputación de frecuencias.
- Modelado paramétrico usando **ETS** (Exponential Smoothing) y comparativas listas con SARIMA.
- Cálculo predictivo del costo final de los Equipos 1 y 2 usando simulaciones probabilísticas e intervalos de confianza.
- Generación automatizada de artefactos (CSVs y gráficos).

### 2️⃣ `prueba2_rag/`
**Asistente Legal basado en RAG (Retrieval-Augmented Generation)**
Arquitectura para procesamiento de leguaje natural especializado en providencias:
- Limpieza y vectorización TF-IDF inicial, y soporte híbrido con Embeddings Semánticos.
- Indexación eficiente usando **FAISS** (operación 100% off-line) pre-cargada.
- QA Generator con soporte múltiple: llamadas a **Azure OpenAI** (Cloud), **Ollama local** (Edge), y un modelo **Extractivo** (Reglas base).
- Formulación sintética de "respuestas amigables" para usuarios sin formación legal.

---

## 🛠 Prerrequisitos y Configuración

El proyecto está diseñado para funcionar en contenedores asegurando portabilidad.

1. **Docker y Docker Compose V2** (Recomendado).
2. (Alternativo) Entorno Python 3.10+, pip (gestor de dependencias) y entorno virtual.

*Para los enfoques Cloud (Azure OpenAI), se proveen los archivos `.env.example` en cada proyecto que deben ser renombrados a `.env` introduciendo las respectivas credenciales si se quiere probar en vivo la conexión.*

---

## 📋 Resumen de la Ejecución y Rúbricas Evaluadas

Se ha priorizado un código altamente cohesionado:
- **Esfuerzo y Proactividad:** Incorporación de API RESTful (FastAPI), multi-modelos (ETS, SARIMA), despliegue on-premise (Docker), y flujos RAG resilientes.
- **Conocimiento Analítico:** Supresión controlada de gaps temporales en forecast y manejo estocástico para los límites de los contratos.
- **Calidad de Software:** `pytest` integrado para álgebra crítica y procesos automáticos `make/compose`.

*(Para el detalle metodológico exhaustivo, consultar el archivo `informe.md` dentro de la carpeta de cada prueba).*

## Autor

- [森光寛] Cristian Morimitsu
