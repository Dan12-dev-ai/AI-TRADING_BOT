# DEDAN Trading Bot - Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.12-slim as production

# Set labels for metadata
LABEL maintainer="DEDAN Team" \
      org.opencontainers.image.title="DEDAN Trading Bot" \
      org.opencontainers.image.description="Ultimate AI trading bot for 2026 markets" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/dedan/dedan"

# Create non-root user
RUN groupadd -r dedan && \
    useradd -r -g dedan dedan

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=dedan:dedan . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/config && \
    chown -R dedan:dedan /app

# Set permissions
RUN chmod +x /app/medallion_x/main.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Expose ports
EXPOSE 8000

# Switch to non-root user
USER dedan

# Start command
CMD ["python", "-m", "medallion_x.main", "--env", "production"]
