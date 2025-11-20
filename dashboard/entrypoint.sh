#!/bin/bash
set -e

# Determine which app to run based on PORT environment variable
if [ "$PORT" = "8051" ]; then
    echo "Starting Admin Dashboard on port 8051..."
    exec python admin.py
else
    echo "Starting Main Dashboard on port ${PORT:-8050}..."
    exec gunicorn --bind 0.0.0.0:${PORT:-8050} --workers 2 --threads 4 --timeout 120 app:server
fi
