#!/bin/bash
set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "============================================"
echo "Service: Data Collection (Camera Feature Uploader)"
echo "============================================"

# Start Data Collection Service
# Note: Ensure your python file is named 'data_collection.py' inside app/core/orchestrators/
echo "Starting Data Collection Process..."
python3 -m app.core.orchestrators.data_collection > data_collection.log 2>&1 &

echo "============================================"
echo "Process started successfully!"
echo "============================================"
echo "Check logs:"
echo "  - Tail logs: tail -f data_collection.log"
echo ""
echo "Use 'pkill -f python3' to stop the process."

wait