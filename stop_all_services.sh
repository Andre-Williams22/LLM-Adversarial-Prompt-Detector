#!/bin/bash

# Stop All Services Script
# This script stops all running services

echo "🛑 Stopping LLM Adversarial Prompt Detector Services"
echo "===================================================="

# Stop Docker services
echo "🐳 Stopping Docker services..."
docker-compose -f docker-compose.monitoring.yml down

# Kill processes on our ports
echo "🧹 Stopping application processes..."
lsof -ti:8080,5000,9090,3000 | xargs kill -9 2>/dev/null || echo "No processes to kill"

# Clean up any remaining Python processes
pkill -f "python main.py" 2>/dev/null || echo "No main.py processes to kill"
pkill -f "mlflow ui" 2>/dev/null || echo "No MLflow processes to kill"

echo "✅ All services stopped"
