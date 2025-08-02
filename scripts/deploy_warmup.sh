#!/bin/bash

# Deployment warm-up script for CapRover or similar platforms
# This script should be run after deployment to pre-load models

echo "Starting post-deployment warm-up..."

# Wait for the app to be fully deployed
echo "Waiting for app to be ready..."
sleep 30

# Get the app URL (modify this for your deployment)
APP_URL="${APP_URL:-https://safe-prompts.andrewilliams.ai}"

echo "Warming up: $APP_URL"

# Run the warm-up script
python3 warm_up.py "$APP_URL"

if [ $? -eq 0 ]; then
    echo "Deployment warm-up completed successfully!"
else
    echo "Deployment warm-up completed with issues"
fi
