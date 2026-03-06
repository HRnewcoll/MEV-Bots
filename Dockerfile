# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# MEV Bot Dashboard — Docker image
# ---------------------------------------------------------------------------
# Build:  docker build -t mev-bots .
# Run:    docker run -p 5000:5000 mev-bots
# Or:     docker compose up
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for safety
RUN useradd -m -u 1000 mevbot
WORKDIR /app

# Install Python deps first (Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source
COPY --chown=mevbot:mevbot . .

# Runtime directory
RUN mkdir -p simulator_data && chown mevbot:mevbot simulator_data

USER mevbot

# Dashboard port
EXPOSE 5000

# Health check — ping the status endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Launch via run.py (handles path setup, dir creation, etc.)
# --no-browser disables the browser auto-open (irrelevant in a container)
# --host 0.0.0.0 so the port is reachable from outside the container
CMD ["python", "run.py", "--host", "0.0.0.0", "--no-browser", "--skip-install"]
