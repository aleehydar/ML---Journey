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

# Layer 4: Install system dependencies (for sentence-transformers FAISS build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layer 5: Copy requirements FIRST (Docker caches this layer separately)
# This means if you only change app code, Docker won't re-install dependencies!
COPY requirements.txt .

# Layer 6: Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn langchain langchain-groq \
    langchain-community sentence-transformers faiss-cpu python-dotenv \
    langchain-core

# Layer 7: Copy the rest of the application code
COPY . .

# Layer 8: Expose the FastAPI port
EXPOSE 8000

# Layer 9: Start the Uvicorn server
# --host 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
