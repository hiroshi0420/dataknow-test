# Cost Forecast (Time Series)

Proyecto reproducible para estimar y proyectar el costo de **Equipo 1** y **Equipo 2** a partir de tres materias primas (X, Y, Z) con datos históricos en formatos heterogéneos.

## Objetivo
- Normalizar y alinear X/Y/Z a frecuencia mensual (intersección temporal).
- Calcular costos históricos de los equipos.
- Entrenar un modelo de series de tiempo (ETS) y generar forecast a 36 meses con intervalos de confianza.
- Exportar artefactos: CSV + gráficas.
- (Opcional) Servir predicciones vía API local en Docker y/o desplegar un endpoint en **Azure ML Online Endpoints**.

## Estructura
```
Datos/                 # X.csv, Y.csv, Z.csv
notebooks/             # EDA opcional
outputs/               # artefactos del pipeline (csv, png, modelos)
src/                   # código fuente
tests/                 # tests unitarios
azureml/               # despliegue opcional a Azure ML Online Endpoint
main.py                # entrypoint del pipeline
informe.md             # reporte final (para entregar)
```

## Ejecución local
```bash
pip install -r requirements.txt
python main.py --method ets --horizon 36
```

Los resultados quedan en `outputs/`.

## Docker (local)
```bash
docker compose up --build pipeline
# opcional: servicio de API para consultas
docker compose up --build api
```

## Azure ML (opcional)
Ver `azureml/README.md`.
