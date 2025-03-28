#!/bin/bash

echo "📦 Deploying T1 Brain to Production..."

cd /root/projects/t1-brain || exit 1
source brain/bin/activate

echo "✅ Pulling latest from GitHub..."
git pull origin main

echo "✅ Installing dependencies..."
pip install -r requirements.txt

echo "🔄 Restarting FastAPI service..."
sudo systemctl restart t1brain-api

echo "🚀 Deployment complete!"
