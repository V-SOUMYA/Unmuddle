FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UNMUDDLE_EMBEDDER=gemini \
    UNMUDDLE_DB=/tmp/unmuddle.db

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY unmuddle ./unmuddle
COPY data ./data
COPY web ./web

# Most hosts set $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn unmuddle.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
