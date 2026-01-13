# Stage 1: Builder - Install dependencies and compile wheels
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to virtual environment
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with wheel caching
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim AS runtime

# Security: Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup \
    && useradd -m -u ${UID} -g appgroup -s /bin/bash appuser

WORKDIR /app

# Install runtime system dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (order by change frequency for cache optimization)
COPY --chown=appuser:appgroup config.py .
COPY --chown=appuser:appgroup settings.py .
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup evaluation/ ./evaluation/
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup app.py .

# Create directories with correct ownership
RUN mkdir -p outputs static temp logs \
    && chown -R appuser:appgroup /app

# Copy trained model if exists
COPY --chown=appuser:appgroup outputs/ ./outputs/

# Security: Switch to non-root user
USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    APP_LOG_FORMAT=json

# Expose port
EXPOSE 8000

# Health check with separate liveness and readiness probes
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root user with gunicorn for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# Stage 3: Development - Full tooling for local development
FROM runtime AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    flake8 \
    mypy

USER appuser

# Override CMD for development (with reload)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
