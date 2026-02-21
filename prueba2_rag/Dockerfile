FROM python:3.11-slim

WORKDIR /app

# Referencias para el pipeline
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Como defecto se ejecuta el pipeline (se puede exponer API como servicio de consulta)
CMD ["python", "main.py"]
