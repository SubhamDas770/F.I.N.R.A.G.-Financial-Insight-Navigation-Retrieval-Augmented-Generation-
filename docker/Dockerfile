# ==========================================
# Multi-Stage GPU Container for Enterprise RAG
# ==========================================

# Stage 1: Build & Dependencies
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install/deps -r requirements.txt

# Stage 2: Final Runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed python packages from builder stage
COPY --from=builder /install/deps /usr/local

# Copy application source code
COPY src/ /app/src/
COPY requirements.txt /app/
COPY README.md /app/

# Create persistent storage directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/vector_db

EXPOSE 7860 8000

CMD ["python3", "-m", "src.ui.app"]
