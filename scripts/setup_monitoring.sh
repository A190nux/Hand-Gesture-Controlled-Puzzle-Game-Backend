#!/bin/bash

echo "🚀 Setting up Hand Gesture API Monitoring Stack"

# Create monitoring directories
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# Make sure the monitoring module directory exists
mkdir -p app/monitoring

echo "📁 Directory structure created"

# Build and start the monitoring stack
echo "🐳 Building and starting containers..."
docker-compose down
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Test API health
echo "🏥 Testing API health..."
curl -s http://localhost:8000/health | jq .

# Test Prometheus
echo "📊 Testing Prometheus..."
curl -s http://localhost:9090/-/healthy

echo "✅ Monitoring stack setup complete!"
echo ""
echo "🌐 Access URLs:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin123)"
echo "  - Metrics: http://localhost:8000/metrics"
echo ""
echo "📈 To run load test:"
echo "  python tests/load_test.py"
