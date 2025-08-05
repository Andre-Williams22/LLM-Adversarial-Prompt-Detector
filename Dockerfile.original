FROM python:3.11-slim

# Install minimal system dependencies with retry and fix-missing
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --fix-missing \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a simple health check endpoint for deployment
RUN echo '#!/bin/bash\ncurl -f http://localhost:80/health || exit 1' > /health.sh && chmod +x /health.sh

EXPOSE 80

# Use uvicorn with proper host binding for deployment and model preloading
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1", "--timeout-keep-alive", "65"]