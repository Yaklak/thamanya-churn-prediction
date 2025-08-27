# ===== Builder stage =====
FROM python:3.10.13-slim AS builder

# Avoid prompts & set UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# If you sometimes use a private index, you can pass it at build time:
#   docker build --build-arg PIP_INDEX_URL=https://<mirror>/simple .
ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

COPY requirements.txt ./

# Upgrade pip and install requirements
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== Final stage =====
FROM python:3.10.13-slim AS final

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . /app

# Create a non-root user and own the app dir
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Optional: lightweight healthcheck (relies on the server being up)
# We'll check /health, falling back to /model/info if needed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || curl -fsS http://127.0.0.1:8000/model/info || exit 1

# By default, load the "best" model from models/artifacts/best/model.joblib.
# Make sure you have run training locally (make train) and committed or volume-mounted artifacts.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
