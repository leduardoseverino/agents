# Multi-stage Dockerfile for AG2 Enhanced Documentation Flow
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Remove unnecessary files
RUN rm -rf \
    .git \
    .github \
    __pycache__ \
    *.pyc \
    *.pyo \
    *.pyd \
    .Python \
    env \
    venv \
    .venv \
    pip-log.txt \
    pip-delete-this-directory.txt \
    .tox \
    .coverage \
    .coverage.* \
    .cache \
    nosetests.xml \
    coverage.xml \
    *.cover \
    *.log \
    .idea \
    .vscode \
    docs/ \
    logs/ \
    workdir/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create necessary directories
RUN mkdir -p logs docs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]