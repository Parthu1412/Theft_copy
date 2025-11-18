#!/bin/bash
set -e

echo "Starting Theft Detection (warming up model)..."
python3 -m app.core.orchestrators.theft &

echo "Starting Video Generation Process..."
python3 -m app.core.orchestrators.video_gen &

echo "Waiting 10 seconds for theft model to warm up..."
sleep 10

echo "Starting Camera + YOLO Orchestrator..."
python3 -m app.core.orchestrators.camera &

echo "All processes started. Use 'pkill -f python3' to stop all processes."

wait 
