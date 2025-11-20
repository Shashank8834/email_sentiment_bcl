# dashboard/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-dashboard.txt ./

# Install Python dependencies + gunicorn
RUN pip install --no-cache-dir -r requirements-dashboard.txt \
 && pip install --no-cache-dir gunicorn \
 && rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Add healthcheck endpoint to both apps
RUN echo 'from flask import Flask; health_app = Flask(__name__); \
          @health_app.route("/_health"); def health(): return "OK", 200' > health.py

# Create non-root user
RUN useradd -m -u 1000 dashuser && chown -R dashuser:dashuser /app
USER dashuser

# Expose both ports
EXPOSE 8050 8051

# Use the smart entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Healthcheck (works for both apps)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:${PORT:-8050}/_health || exit 1
