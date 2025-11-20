# monitor/Dockerfile - Updated for PostgreSQL support
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including PostgreSQL client libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Increase pip network timeout (helps with flaky networks)
ENV PIP_DEFAULT_TIMEOUT=300

COPY requirements-monitor.txt /app/

# Upgrade pip and install requirements
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements-monitor.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Copy application files including db_config
COPY graph_delegate_monitor.py inference_local.py db_config.py /app/

# Environment variables (DB_PATH kept for backward compatibility)
ENV DB_PATH=/data/monitor.db
ENV INFERENCE_MODEL_DIR=/data/model

CMD ["python", "graph_delegate_monitor.py"]