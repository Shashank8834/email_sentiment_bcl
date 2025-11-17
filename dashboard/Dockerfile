# dashboard/Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# small system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# NOTE: build context is ./dashboard so requirements-dashboard.txt must be in dashboard/
COPY requirements-dashboard.txt /app/requirements-dashboard.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements-dashboard.txt

# Copy dashboard source from context (./dashboard) into container
COPY . /app

# Ensure data directory (mounted at runtime) exists
RUN mkdir -p /data && chmod 777 /data

# Default command
CMD ["python", "app.py"]
