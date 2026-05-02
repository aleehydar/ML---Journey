# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a temporary wheel directory
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt


# Stage 2: Runtime
FROM python:3.10-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy the built wheels from the builder stage
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install packages from wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY ./app /app/app

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Create cache directory with correct permissions for non-root user
RUN mkdir -p /tmp/hf_cache && chmod 777 /tmp/hf_cache

# Switch to non-root user
USER appuser

# Expose port 8001
EXPOSE 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
