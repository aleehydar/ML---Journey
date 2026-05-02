# ─────────────────────────────────────────────────────────────────────────────
# Pakistan Legal Intelligence — Dockerfile
#
# Builds a single production container running the FastAPI server on port 8000.
# Usage:
#   docker build -t pakistan-legal-ai .
#   docker run -p 8000:8000 --env GROQ_API_KEY=your_key pakistan-legal-ai
# ─────────────────────────────────────────────────────────────────────────────

# Layer 1: Use a minimal, fast Python 3.11 base image
FROM python:3.11-slim

# Layer 2: Set environment variables to prevent .pyc files and buffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Layer 3: Set working directory inside the container
WORKDIR /app

# Layer 4: Install system dependencies & Syft for SBOM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

# Layer 5: Create non-root user and set permissions
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /app/data /app/vectorstore /app/monitoring && \
    chown -R appuser:appuser /app

# Layer 6: Copy requirements FIRST (Docker caches this layer separately)
COPY requirements.txt .

# Layer 7: Install Python dependencies
# Pre-install PyTorch explicitly targeting CPU to bypass multi-gigabyte CUDA downloads
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Layer 8: Download Syft to generate SBOM (optional artifact step in build)
RUN curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Layer 9: Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Generate an SBOM for audit
RUN syft dir:. -o cyclonedx-json=bom.json && chown appuser:appuser bom.json

# Layer 10: Switch to non-root user
USER appuser

# Layer 11: Expose the FastAPI port
EXPOSE 8000

# Layer 12: Start the Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
