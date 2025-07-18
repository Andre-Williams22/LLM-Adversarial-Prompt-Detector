#!/bin/bash

# Script to update Grafana dashboard from JSON file
# Usage: ./update-dashboard.sh

echo "🔄 Updating Grafana dashboard..."

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Check if Grafana is running
if ! curl -s http://localhost:3000 &> /dev/null; then
    echo "❌ Error: Grafana is not running on port 3000. Please start Grafana first."
    exit 1
fi

# Check if dashboard JSON file exists
if [ ! -f "grafana/dashboards/adversarial-detector-dashboard.json" ]; then
    echo "❌ Error: Dashboard JSON file not found."
    exit 1
fi

# Wrap the dashboard JSON in the required format for Grafana API
jq '{dashboard: ., overwrite: true}' grafana/dashboards/adversarial-detector-dashboard.json > /tmp/dashboard-wrapped.json

# Import the dashboard
response=$(curl -s -X POST -H "Content-Type: application/json" -u admin:admin123 \
  -d @/tmp/dashboard-wrapped.json \
  http://localhost:3000/api/dashboards/db)

echo "📊 Response: $response"

# Check if the update was successful
if echo "$response" | jq -e '.status == "success"' > /dev/null; then
    echo "✅ Dashboard update successful!"
    echo "🌐 View at: http://localhost:3000/d/adversarial-prompt-detector"
else
    echo "❌ Dashboard update failed. Check the response above for details."
    exit 1
fi

# Clean up
rm /tmp/dashboard-wrapped.json

echo "🎯 Dashboard is now updated with your latest changes!"
