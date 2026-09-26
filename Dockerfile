FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GATEWAY_EMBEDDING_MODEL_DIR=/app/models/all-MiniLM-L6-v2
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers start without network access.
COPY gateway/cache/embedder.py /tmp/embedder.py
RUN python /tmp/embedder.py download /app/models/all-MiniLM-L6-v2 && rm /tmp/embedder.py

COPY gateway ./gateway
COPY worker ./worker
COPY mock_provider ./mock_provider
COPY deploy/config ./deploy/config

RUN useradd --system --uid 10001 app && chown -R app /app
USER app

EXPOSE 8080
CMD ["python", "-m", "gateway.serve"]
