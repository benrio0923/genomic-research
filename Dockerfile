# Multi-stage Dockerfile for genomic-research
# Supports CPU and GPU (NVIDIA CUDA) environments

# ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY genomic_research/ genomic_research/

RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage (CPU) ----
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local
COPY genomic_research/ genomic_research/
COPY pyproject.toml README.md LICENSE ./

RUN mkdir -p /root/.cache/genomic-research /app/workspace

WORKDIR /app/workspace

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD genomic-research status || exit 1

ENTRYPOINT ["genomic-research"]
CMD ["--help"]

# ---- GPU stage (NVIDIA CUDA) ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY genomic_research/ genomic_research/

RUN pip install --no-cache-dir --break-system-packages . && \
    pip install --no-cache-dir --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124

RUN mkdir -p /root/.cache/genomic-research /app/workspace

WORKDIR /app/workspace

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD genomic-research status || exit 1

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["genomic-research"]
CMD ["--help"]
