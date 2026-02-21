# Informe – Prueba Técnica 2
## Asesor Legal para Consulta Historia de Demandas

**DataKnow SAS | Científico de Datos**

---

## 1. Explicación del Caso

Un consultorio legal necesita automatizar la consulta de su historial de demandas y sentencias para agilizar la asesoría a clientes. Actualmente los abogados consultan manualmente un Excel con 329 casos históricos para orientar a sus clientes sobre posibles resultados de nuevas demandas.

El reto específico de esta prueba es crear una PoC (Prueba de Concepto) de IA Generativa que permita hacer preguntas en lenguaje natural y recibir respuestas claras, en lenguaje coloquial, sin jerga jurídica, respaldadas en los casos reales del archivo.

---

## 2. Supuestos

- El Excel contiene 329 filas con columnas: `Providencia`, `Fecha Sentencia`, `Tema - subtema`, `resuelve` y `sintesis`. Se verificó que existen casos sobre redes sociales (47), acoso escolar (9) y PIAR (2).
- Las respuestas deben generarse **solo** con base en el contenido del Excel; el sistema no puede inventar información.
- Se asume que los clientes no tienen conocimientos legales, por lo que el lenguaje debe ser completamente accesible.
- Para las 3 demandas de redes sociales pedidas, se eligieron automáticamente las más relevantes semánticamente mediante retrieval; esta selección es reproducible y auditable.
- Los precios de materias primas y las monedas no aplican en esta prueba.

---

## 3. Formas para Resolver el Caso y Opción Tomada

### Opciones evaluadas

**Opción A – Búsqueda por keywords (determinística)**
Filtrar el Excel por palabras clave y devolver las filas. Simple, rápido, sin LLM.
*Limitación:* No entiende preguntas en lenguaje natural ni sintetiza respuestas coloquiales.

**Opción B – Fine-tuning de un LLM**
Entrenar un modelo sobre los 329 casos para que "memorice" las sentencias.
*Limitación:* Costoso, lento, propenso a alucinaciones, y los 329 casos son pocos para fine-tuning robusto.

**Opción C – RAG (Retrieval-Augmented Generation) ← Opción tomada**
Combina recuperación semántica con generación controlada:
1. Convertir cada caso en un documento de texto estructurado.
2. Generar embeddings y almacenarlos en un índice FAISS.
3. Para cada pregunta: recuperar los casos más relevantes.
4. Pasar los casos recuperados al LLM como contexto, no como memoria.
5. El LLM responde solo con lo que está en el contexto.

**¿Por qué RAG?**
- Evita alucinaciones: el LLM solo puede responder con lo que se le pasa.
- Escalable: si el Excel crece, solo hay que re-indexar.
- Auditable: cada respuesta incluye la providencia fuente.
- Sin necesidad de fine-tuning ni datos de entrenamiento adicionales.

### Mejora adicional: filtro determinístico + retrieval semántico

Para preguntas sobre temas específicos (redes sociales, acoso escolar, PIAR) se aplica primero un filtro por keywords antes del retrieval semántico. Esto reduce el espacio de búsqueda y aumenta la precisión, evitando que documentos irrelevantes entren al LLM.

---

## 4. Resultados del Análisis de los Datos y los Modelos

### Análisis del dataset

| Métrica | Valor |
|---|---|
| Total de casos | 329 |
| Casos sobre redes sociales | 47 |
| Casos sobre acoso escolar/bullying | 9 |
| Casos sobre PIAR | 2 |
| Columnas principales | Providencia, Fecha, Tema, Síntesis, Resuelve |

El campo `Tema - subtema` tiene valores nulos en varias filas, por lo que el sistema usa adicionalmente el campo `sintesis` para la búsqueda y el filtrado.

### Modelo de embeddings

- **Modo local:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) — multilingüe, funciona bien en español.
- **Modo Azure:** `text-embedding-ada-002` vía Azure OpenAI — mayor calidad, requiere API key.

### Vector store

FAISS con `IndexFlatIP` (producto interno = similitud coseno con vectores normalizados). Rápido para 329 documentos, sin necesidad de cuantización.

### Respuestas a las 5 preguntas

*(Las respuestas finales se generan en ejecución y se guardan en `outputs/respuestas.md`. Ver ese archivo para el resultado completo con fuentes.)*

---

## 5. Futuros Ajustes o Mejoras

**Técnicas:**
- Usar un modelo de embeddings más potente (multilingual-e5-large) para mejor recuperación en español jurídico.
- Implementar re-ranking (cross-encoder) sobre los top-k recuperados para mayor precisión.
- Agregar memoria conversacional para preguntas de seguimiento ("¿y en ese caso, cuánto tiempo duró el proceso?").
- Evaluar la calidad de las respuestas con un framework de evaluación RAG (RAGAS o similar).

**Producto:**
- Interfaz web con Streamlit para que los abogados consulten sin línea de comandos.
- Sistema de retroalimentación: el abogado puede marcar si la respuesta fue útil.
- Logging de consultas para análisis de los temas más frecuentes.

**Operación:**
- Automatizar la actualización del índice cuando se agreguen nuevos casos al Excel.
- Despliegue en Azure Container Apps para disponibilidad continua.

---

## 6. Apreciaciones y Comentarios (Opcional)

El caso es un excelente ejemplo de cómo la IA Generativa puede aportar valor inmediato en procesos de conocimiento intensivo sin necesidad de un desarrollo complejo. Un RAG bien construido sobre datos de calidad supera en practicidad a soluciones más complejas como el fine-tuning.

El mayor riesgo en este tipo de sistemas es la alucinación (que el modelo invente sentencias). La solución implementada lo mitiga con tres capas: (1) filtro determinístico previo al retrieval, (2) prompt que prohíbe responder fuera del contexto, y (3) citas obligatorias de fuente en cada respuesta.

Para un entorno real de consultorio legal se recomendaría además una capa de revisión humana ("human in the loop") antes de compartir cualquier respuesta con el cliente final.