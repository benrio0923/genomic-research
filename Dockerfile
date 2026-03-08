FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install package
COPY pyproject.toml README.md LICENSE ./
COPY genomic_research/ genomic_research/
RUN pip install --no-cache-dir -e ".[all]"

# Default: show help
CMD ["genomic-research", "--help"]
