#!/bin/bash
set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default values for configuration flags
NUM_YOLO_WORKERS=${NUM_YOLO_WORKERS:-1}
ENABLE_HEATMAP=${ENABLE_HEATMAP:-false}
ENABLE_PEOPLE_COUNTING=${ENABLE_PEOPLE_COUNTING:-false}
# New optional flag to run camera feature uploader (people/heatmap snapshots + re-id video segments)
ENABLE_FEATURE_UPLOADER=${ENABLE_FEATURE_UPLOADER:-false}

echo "============================================"
echo "Service Configuration:"
echo "  NUM_YOLO_WORKERS: $NUM_YOLO_WORKERS"
echo "  ENABLE_HEATMAP: $ENABLE_HEATMAP"
echo "  ENABLE_PEOPLE_COUNTING: $ENABLE_PEOPLE_COUNTING"
echo "  ENABLE_FEATURE_UPLOADER: $ENABLE_FEATURE_UPLOADER"
echo "============================================"

echo "Starting Theft Detection (warming up model)..."
python3 -m app.core.orchestrators.theft 2>&1 &

#PARTHU
echo "Starting Video Generation Process..."
python3 -m app.core.orchestrators.video_gen 2>&1 &
#PARTHU

# Start Heatmap Service if enabled
if [ "$ENABLE_HEATMAP" = "true" ]; then
    echo "Starting Heatmap Service..."
    python3 -m app.core.orchestrators.heatmap 2>&1 &
    sleep 1
else
    echo "Heatmap Service: DISABLED"
fi

# Start People Counting Service if enabled
if [ "$ENABLE_PEOPLE_COUNTING" = "true" ]; then
    echo "Starting People Counting Service..."
    python3 -m app.core.orchestrators.people_count 2>&1 &
    sleep 1
else
    echo "People Counting Service: DISABLED"
fi

# Start Camera Feature Uploader if enabled
if [ "$ENABLE_FEATURE_UPLOADER" = "true" ]; then
    echo "Starting Camera Feature Uploader..."
    python3 -m app.core.orchestrators.camera_feature_uploader 2>&1 &
    sleep 2
else
    echo "Camera Feature Uploader: DISABLED"
fi

echo "Waiting 40 seconds for theft model to warm up..."
sleep 40

# Start Camera Orchestrator (handles both single and multi-YOLO)
echo "Starting Camera Orchestrator with $NUM_YOLO_WORKERS YOLO worker(s)..."
python3 -m app.core.orchestrators.camera_multi_yolo 2>&1 &

echo "============================================"
echo "All processes started successfully!"
echo "============================================"
echo "Check logs:"
echo "  - Theft: tail -f theft.log"
[ "$ENABLE_HEATMAP" = "true" ] && echo "  - Heatmap: tail -f heatmap.log"
[ "$ENABLE_PEOPLE_COUNTING" = "true" ] && echo "  - People Counting: tail -f people_count.log"
echo "  - Camera: tail -f camera.log"
[ "$ENABLE_FEATURE_UPLOADER" = "true" ] && echo "  - Feature Uploader: tail -f camera_feature_uploader.log"
echo ""
echo "Use 'pkill -f python3' to stop all processes."

wait 
