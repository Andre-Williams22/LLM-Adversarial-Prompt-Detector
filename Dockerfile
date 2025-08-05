# Cloud Run Dockerfile
# Optimized for fast startup and efficient resource usage

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (exclude BentoML for faster install)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download models and cache them to reduce startup time
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Create necessary directories
RUN mkdir -p data/preprocessed logs

# Set environment variables for production
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application with optimized startup
CMD ["python", "-u", "startup_cloudrun.py"]
