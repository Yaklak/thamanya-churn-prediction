# Use slim Python image
FROM python:3.11-slim

# System deps (optional but handy for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Copy ONLY requirements first to leverage Docker layer caching
COPY requirements.txt .

# 2) Install Python deps (upgrade pip first)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3) Copy the rest of your project
COPY . .

# Env hygiene
ENV PYTHONUNBUFFERED=1

# Default command: serve the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]