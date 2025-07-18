#!/bin/bash

# Start Grafana and Proecho "📈 Your App Metrics: http://localhost:8001/metrics"etheus monitoring stack
echo "Starting Grafana and Prometheus monitoring stack..."

# Try docker compose first (newer Docker versions), fallback to docker-compose
if command -v docker &> /dev/null; then
    if docker compose version &> /dev/null; then
        echo "Using 'docker compose' command..."
        docker compose -f docker-compose.monitoring.yml up -d
    elif command -v docker-compose &> /dev/null; then
        echo "Using 'docker-compose' command..."
        docker-compose -f docker-compose.monitoring.yml up -d
    else
        echo "❌ Error: Neither 'docker compose' nor 'docker-compose' command found!"
        echo "Please install Docker and Docker Compose first."
        exit 1
    fi
else
    echo "❌ Error: Docker is not installed or not in PATH!"
    echo "Please install Docker first."
    exit 1
fi

echo "Waiting for services to start..."
sleep 10

echo "✅ Monitoring stack started!"
echo ""
echo "🔗 Access URLs:"
echo "📊 Grafana Dashboard: http://localhost:3000"
echo "   - Username: admin"
echo "   - Password: admin123"
echo ""
echo "🔍 Prometheus: http://localhost:9090"
echo ""
echo "📈 Your App Metrics: http://localhost:8080/metrics"
echo ""
echo "To stop the monitoring stack:"
echo "docker compose -f docker-compose.monitoring.yml down"
echo "or"
echo "docker-compose -f docker-compose.monitoring.yml down"
