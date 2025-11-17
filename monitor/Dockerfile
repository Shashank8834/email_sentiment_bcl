# monitor/Dockerfile (uses CPU-only PyTorch wheels & increases pip timeout)
FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# increase pip network timeout (helps flaky networks)
ENV PIP_DEFAULT_TIMEOUT=300

COPY requirements-monitor.txt /app/

# upgrade pip and install lighter requirements first, then install CPU torch from official index
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements-monitor.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY graph_delegate_monitor.py inference_local.py /app/

ENV DB_PATH=/data/monitor.db
ENV INFERENCE_MODEL_DIR=/data/model

CMD ["python", "graph_delegate_monitor.py"]
