#!/bin/bash

# Start All Services Script
# This script starts the main app, MLflow, and monitoring services on different ports

echo "🚀 Starting LLM Adversarial Prompt Detector Services"
echo "=================================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate virtual environment
source env/bin/activate

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8080,5000,9090,3000 | xargs kill -9 2>/dev/null || echo "No existing processes to kill"

# Start MLflow UI
echo "📈 Starting MLflow UI on port 5000..."
mkdir -p logs
mlflow ui --host 0.0.0.0 --port 5000 > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"

# Start monitoring services (Prometheus & Grafana)
echo "📊 Starting monitoring services..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait a moment for services to start
sleep 3

# Start the main application
echo "🤖 Starting main application on port 8080..."
mkdir -p logs
python main.py > logs/app.log 2>&1 &
APP_PID=$!
echo "Main app PID: $APP_PID"

# Wait for startup
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "🔍 Checking service status..."
echo "================================"

# Check main app
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Main App: http://localhost:8080 (RUNNING)"
    echo "   - Chat Interface: http://localhost:8080/chat"
    echo "   - API Health: http://localhost:8080/health"
    echo "   - Metrics: http://localhost:8080/metrics"
else
    echo "❌ Main App: http://localhost:8080 (FAILED)"
fi

# Check MLflow
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ MLflow UI: http://localhost:5000 (RUNNING)"
else
    echo "❌ MLflow UI: http://localhost:5000 (FAILED)"
fi

# Check Prometheus
if curl -s http://localhost:9090 > /dev/null; then
    echo "✅ Prometheus: http://localhost:9090 (RUNNING)"
else
    echo "❌ Prometheus: http://localhost:9090 (FAILED)"
fi

# Check Grafana
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Grafana: http://localhost:3000 (RUNNING)"
    echo "   - Username: admin"
    echo "   - Password: admin123"
else
    echo "❌ Grafana: http://localhost:3000 (FAILED)"
fi

echo ""
echo "🎯 Quick Access Links:"
echo "================================"
echo "Chat Interface: http://localhost:8080/chat"
echo "MLflow UI: http://localhost:5000"
echo "Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "Prometheus: http://localhost:9090"
echo "Health Check: http://localhost:8080/health"

echo ""
echo "📝 Process IDs:"
echo "Main App: $APP_PID"
echo "MLflow: $MLFLOW_PID"
echo "Docker services: Use 'docker-compose -f docker-compose.monitoring.yml ps'"

echo ""
echo "🛑 To stop all services, run: ./stop_all_services.sh"
echo "📋 To view logs: tail -f logs/app.log (or logs/mlflow.log)"
