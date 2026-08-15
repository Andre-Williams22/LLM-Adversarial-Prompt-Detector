#!/bin/bash

# Stop All Services Script
# This script stops all running services

echo "Stopping LLM Adversarial Prompt Detector Services"
echo "===================================================="

# Stop Docker services
echo "Stopping Docker services..."
docker-compose -f monitoring/docker-compose.yml down

# Kill processes on our ports
echo "Stopping application processes..."
lsof -ti:8080,5000,9090,3000 | xargs kill -9 2>/dev/null || echo "No processes to kill"

# Clean up any remaining Python processes
pkill -f "uvicorn main:app" 2>/dev/null || echo "No application processes to kill"
pkill -f "mlflow ui" 2>/dev/null || echo "No MLflow processes to kill"

echo "All services stopped"
