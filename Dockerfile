# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# Avoid interactive prompts & ensure UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System dependencies only (no build toolchain needed for manylinux wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optionally allow private index at build time
ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Pre-install pip and cache wheels to speed up rebuilds
RUN python -m pip install --upgrade pip

# Install runtime dependencies first (better Docker layer caching)
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Healthcheck for FastAPI
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || curl -fsS http://127.0.0.1:8000/model/info || exit 1

# Default command: serve the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
