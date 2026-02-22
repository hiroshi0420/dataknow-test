# Informe – Estimación de Costos con Series de Tiempo
## Estimación de Costos de Equipos para Proyecto de Construcción

**DataKnow SAS | Científico de Datos**

---

## 1. Explicación del Caso

Una empresa constructora planifica un proyecto de **36 meses** y necesita estimar el costo de dos equipos esenciales cuyos precios dependen de materias primas con cotización variable en el mercado.

Las fórmulas de composición de precio son:
- **Equipo 1** = 20% × Materia Prima X + 80% × Materia Prima Y
- **Equipo 2** = (X + Y + Z) / 3 *(partes iguales)*

El objetivo es generar una estimación confiable del costo de ambos equipos durante los 36 meses del proyecto, que sirva como insumo para la planeación financiera y la negociación con proveedores.

---

## 2. Supuestos

- **Moneda y unidad:** Se asume que X, Y y Z están expresados en la misma moneda y unidad de medida. El análisis es relativo (tendencia y variación), no absoluto en términos monetarios.
- **Frecuencia de análisis:** Se trabajó en frecuencia **mensual** mediante promedio de los precios diarios. Justificación: (1) el proyecto de planificación es mensual, (2) reduce el ruido de precios diarios, (3) es la unidad natural para contratos de largo plazo.
- **Rango de trabajo:** Se usó la **intersección** de fechas donde existen datos en los tres CSVs simultáneamente: **enero 2010 → agosto 2023** (164 meses). Esto evita imputar precios ficticios donde no hay observaciones reales.
- **Datos de los CSV:** Los tres archivos tienen formatos distintos que requirieron normalización específica:
  - `X.csv`: separador coma, fechas ISO, precio con punto decimal *(más limpio)*
  - `Y.csv`: separador punto y coma, fechas `DD/M/YYYY`, precio con **coma decimal** *(más complejo)*
  - `Z.csv`: columnas **invertidas** (Price antes que Date), fechas ISO *(requiere reordenamiento)*
- **Horizonte de forecast:** 36 meses hacia adelante desde agosto 2023 (septiembre 2023 → agosto 2026), que corresponde a la duración del proyecto.
- **Independencia de las materias primas:** Para el cálculo de intervalos de confianza del equipo, se asume independencia entre X, Y y Z (supuesto conservador; en la práctica pueden estar correlacionadas).

---

## 3. Formas para Resolver el Caso y Opción Tomada

### Opciones evaluadas

**Opción A – Precio histórico promedio**
Calcular el precio promedio del histórico como estimación puntual para los 36 meses.
*Limitación:* No captura tendencia ni estacionalidad. Si los precios están en una tendencia alcista, subestima el costo futuro. No da incertidumbre.

**Opción B – Precio del último mes conocido (random walk)**
Usar el último precio observado como estimación constante para el futuro.
*Limitación:* Es el baseline más simple (naive); útil como referencia pero no aprovecha los patrones históricos.

**Opción C – Modelo de series de tiempo ETS ← Opción tomada**
Exponential Smoothing (ETS) con tendencia aditiva y estacionalidad multiplicativa.
Ventajas:
- Captura tendencia a largo plazo y patrones estacionales
- Interpretable: parámetros con significado claro
- Funciona bien en series de precios de commodities
- Entrega intervalos de confianza mediante simulación bootstrap

**Opción D – ARIMA / SARIMA (alternativa implementada)**
Auto-ARIMA para selección automática de parámetros.
*Cuándo usar:* Si la serie tiene patrones no lineales o la ETS no pasa diagnósticos. Implementado como alternativa activable con `--method arima`.

### Estrategia de forecast adoptada

Se forecastearon **X, Y y Z por separado** y luego se aplicaron las fórmulas de los equipos sobre los forecasts. Este enfoque es más correcto que forecastear los equipos directamente porque:
1. Cada materia prima tiene su propia dinámica de mercado
2. Permite sensibilidad: ¿qué pasa si solo X sube?
3. Los intervalos de confianza se propagan con las fórmulas originales

Se realizó **backtesting rolling** sobre los últimos 12 meses del histórico para evaluar la calidad del modelo antes de proyectar.

---

## 4. Resultados del Análisis de los Datos y los Modelos

### Estadísticas descriptivas del histórico (2010-01 → 2023-08)

*(Los valores exactos se generan al correr el pipeline y se guardan en `outputs/estadisticas_descriptivas.csv`)*

| Serie | Media | Desv. Est. | CV (%) | Var. Total (%) |
|---|---:|---:|---:|---:|
| X | 78.07 | 25.14 | 32.2 | 10.5 |
| Y | 555.50 | 138.09 | 24.9 | 1.9 |
| Z | 2037.61 | 370.50 | 18.2 | -4.5 |
| Equipo 1 | 460.01 | 113.05 | 24.6 | 2.2 |
| Equipo 2 | 890.39 | 168.87 | 19.0 | -2.9 |

> Valores calculados por el pipeline (`outputs/estadisticas_descriptivas.csv`).

### Observaciones del EDA

- **X** presenta la mayor volatilidad relativa (**CV = 32.2%**), seguida de **Y** (24.9%) y **Z** (18.2%).
- **Y** tiene la mayor influencia en Equipo 1 (80% del peso), por lo que la estabilidad de Y es crítica para la planificación.
- **Z** tiene la serie más larga y estable de las tres.
- Los costos de **Equipo 1 y Equipo 2 muestran tendencias distintas** debido a la diferente composición: Equipo 1 está dominado por Y, mientras Equipo 2 es más sensible a Z por su magnitud.

### Backtesting (últimos 12 meses)

*(Los valores exactos se generan al correr `main.py`)*

| Equipo | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| Equipo 1 | 22.48 | 26.43 | 4.7 |
| Equipo 2 | 32.65 | 40.60 | 3.3 |

### Forecast 36 meses

*(Los valores exactos se guardan en `outputs/forecast_36m.csv`)*

| Punto del proyecto | Equipo 1 (media) | IC 95% | Equipo 2 (media) | IC 95% |
|---|---:|---:|---:|---:|
| Mes 1 (sep 2023) | 458.08 | [406.39, 513.23] | 926.70 | [835.98, 1017.51] |
| Mes 12 (ago 2024) | 436.53 | [211.41, 673.26] | 900.33 | [550.17, 1283.16] |
| Mes 36 (ago 2026) | 385.43 | [-113.78, 962.33] | 851.21 | [-36.19, 1815.67] |

**Recomendación para contratos:** Usar el **percentil 90 (IC 95% superior)** como presupuesto conservador para negociación con proveedores. Esto cubre el 95% de los escenarios posibles con el modelo.

---

## 5. Futuros Ajustes o Mejoras

**Técnicas:**
- Evaluar correlación entre X, Y, Z para propagar correctamente la incertidumbre del equipo (covarianza, no independencia).
- Incorporar variables exógenas: índices de precios industriales, tipo de cambio, eventos de mercado.
- Probar modelos de conjunto (ensemble) para forecast más robusto.
- Análisis de escenarios explícitos: shock alcista/bajista en una sola materia prima.

**Negocio:**
- Sensibilidad: ¿cuánto cambia el costo del equipo si el mix 20/80 de Equipo 1 varía?
- Comparación proveedores: con la curva de forecast como referencia, evaluar si un proveedor que ofrece precio fijo está por encima o debajo del percentil esperado.
- Alertas de precios: monitoreo mensual con aviso si el precio real sale del IC previsto.

---

## 6. Apreciaciones y Comentarios (Opcional)

El caso ilustra un desafío clásico en consultoría: los datos llegan con formatos inconsistentes y rangos distintos, y la mayor parte del trabajo es la normalización previa al análisis. En este caso, Y.csv requirió manejo especial (separador punto y coma + coma decimal), Z.csv tenía las columnas invertidas, y los tres archivos cubren rangos temporales distintos.

La elección de trabajar con la intersección (vs imputación) es conservadora pero honesta: evita "inventar" precios donde no existe información real. En un contexto real de consultoría, esta decisión se discutiría con el cliente antes de proceder.
