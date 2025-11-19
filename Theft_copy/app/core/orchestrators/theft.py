#!/usr/bin/env python3
"""
Theft Detection Processor
"""

import cv2
import numpy as np
import zmq, logging
import os, sys, time
import tensorflow as tf
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

load_dotenv()

from app import config
from app.core.services.inference import TheftInferenceService
from app.core.services.api_service import TheftAPIService

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("theft_detection")


class TheftCameraState:
    __slots__ = ("frame_count", "waiting", "trigger_prob")

    def __init__(
        self,
        frame_count: int = 0,
        waiting: bool = False,
    ):
        self.waiting = waiting
        self.frame_count = frame_count
        self.trigger_prob = 0.0

    def stop_waiting(self):

        self.waiting = False
        self.trigger_prob = 0.0

    def start_waiting(self, theft_prob: float):

        self.waiting = True
        self.trigger_prob = theft_prob  # Store the triggering probability
        logger.info(f"THEFT DETECTION: Started waiting for 300 frames with prob {theft_prob:.4f}...")


class TheftOrchestrator:
    """Main orchestrator for theft detection."""

    def __init__(self):
        """Initialize the orchestrator."""

        self.context = zmq.Context()

        # Create receivers for all YOLO workers
        self.receivers = []
        num_workers = config.NUM_YOLO_WORKERS
        for worker_id in range(num_workers):
            receiver = self.context.socket(zmq.PULL)
            port = 5590 + worker_id
            receiver.bind(f"tcp://*:{port}")
            receiver.setsockopt(zmq.RCVHWM, 300)
            self.receivers.append(receiver)
            logger.info(f"Theft Orchestrator: Bound to port {port} for YOLO worker {worker_id}")

        #PARTHU
        # Video generation sender
        self.video_sender = self.context.socket(zmq.PUSH)
        self.video_sender.connect("tcp://localhost:5559")
        self.video_sender.setsockopt(zmq.SNDHWM, 10)
        self.video_sender.setsockopt(zmq.LINGER, 0) 
        #logger.info("Theft Orchestrator: Connected to video generator on tcp://localhost:5559")
        #PARTHU
        
        self.theft_model = TheftInferenceService()
        self.theft_model.warm_up()

        # Initialize API service for sending alerts
        self.api_service = TheftAPIService()

        self.theft_state: Dict[str, TheftCameraState] = {}

        # Theft detection parameters - sliding window size
        self.window_size = config.THEFT_WINDOW_SIZE

        # Tensor buffer for each camera
        self.camera_tensor_buffers: Dict[str, deque] = {}
        self.theft_inference_batch = config.THEFT_INFERENCE_BATCH

        # Original frames buffer for each camera
        self.orig_frames_data: Dict[str, deque] = {}
        self.raw_frame_queue_size = config.THEFT_FRAMES_BUFFER

        logger.info("Theft Detection Processor: Initialized")

    def get_state(self, camera_id: str) -> TheftCameraState:

        if camera_id not in self.theft_state:

            self.theft_state[camera_id] = TheftCameraState()

            self.orig_frames_data[camera_id] = deque(maxlen=self.raw_frame_queue_size)

            self.camera_tensor_buffers[camera_id] = deque(
                maxlen=self.theft_inference_batch
            )

        return self.theft_state[camera_id]

    def convert_frame_to_tensor(self, frame: np.ndarray) -> tf.Tensor:

        try:

            if not isinstance(frame, tf.Tensor):
                frame_tf = tf.cast(tf.convert_to_tensor(frame), dtype=tf.float32)
            else:
                frame_tf = tf.cast(frame, dtype=tf.float32)

            return frame_tf

        except Exception as e:
            logger.error(f"Error converting frame to tensor: {e}")

            return tf.zeros((224, 224, 3), dtype=tf.float32)

    def apply_background_mask(
        self, frame: np.ndarray, detections: List[List[int]]
    ) -> np.ndarray:

        try:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for detection in detections:
                if len(detection) >= 4:
                    x1, y1, x2, y2 = detection[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            return masked_frame

        except Exception as e:
            logger.error(f"Error applying background mask: {e}")

            return frame

    def trigger_alert(self, camera_id: str, theft_prob: float, timestamp: float):
        """Trigger theft alert logic - sends alert to API."""

        logger.info(f"ALERT: Theft alert triggered for camera {camera_id}!")

        # # Send API alert
        # try:
        #     # Convert timestamp to unix time for API
        #     from datetime import datetime
        #     if isinstance(timestamp, str):
        #         # Parse ISO format timestamp
        #         ts_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        #         end_time = int(ts_obj.timestamp())
        #     else:
        #         # Already a unix timestamp
        #         end_time = int(timestamp)
            
         
        #     start_time=end_time - 20
            
        #     # Submit theft alert to API
        #     self.api_service.submit_theft_alert(
        #         camera_id=camera_id,
        #         theft_probability=theft_prob,
        #         start_time=start_time,
        #         end_time=end_time
        #     )
        #     logger.info(f"API alert submitted for camera {camera_id} (probability: {theft_prob:.4f})")
            
        #     # Clear the frame buffer after alert is sent
            
            
        # except Exception as e:
        #     logger.error(f"Failed to submit API alert for camera {camera_id}: {e}")
            #PARTHU
        try:
            # Collect original frames and metadata
            orig_frames = [fd["frame"] for fd in self.orig_frames_data[camera_id]]
            frame_count = len(orig_frames)
            store_id = self.orig_frames_data[camera_id][0].get("store_id", "unknown") if frame_count > 0 else "unknown"

            # Normalize timestamp to ISO-8601 with 'Z'
            from datetime import datetime, timezone
            if isinstance(timestamp, str):
                iso_ts = timestamp if timestamp.endswith("Z") else f"{timestamp}Z"
            else:
                iso_ts = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")

            video_data = {
                "camera_id": camera_id,
                "frames": orig_frames,    
                "theft_score": theft_prob,   
                "timestamp": iso_ts,          
                "frame_count": frame_count,
                "store_id": store_id,
            }

            # Non-blocking send; drop if downstream is busy
            try:
                self.video_sender.send_pyobj(video_data, zmq.NOBLOCK)
                logger.info(f"Queued video for camera {camera_id}: {frame_count} frames")
            except zmq.Again:
                logger.warning(f"Video queue full, dropping video for camera {camera_id}")

            
        except Exception as e:
            logger.error(f"Failed to queue video for camera {camera_id}: {e}")
            #PARTHU
        #clear the frame buffer after attempting to send
        self.orig_frames_data[camera_id].clear()
            
        

    def handle_theft_alert_trigger_logic(
        self, camera_id: str, should_trigger: bool, theft_prob: float, timestamp: float
    ):
        """Check if we should trigger an alert based on the model prediction and state."""

        state = self.get_state(camera_id)

        # Trigger alert if model indicates or previously indicated but waiting for more frames
        if (should_trigger or state.waiting) and len(
            self.orig_frames_data[camera_id]
        ) >= self.raw_frame_queue_size:

            # Use stored trigger probability if waiting, otherwise current probability
            prob_to_use = state.trigger_prob if state.waiting else theft_prob

            # Go ahead and trigger the alert
            self.trigger_alert(camera_id, prob_to_use, timestamp)

            # Reset waiting state
            state.stop_waiting()

        elif should_trigger:
            state.start_waiting(theft_prob)


    def process_camera_batch(
        self, camera_id: str, last_batch: List[Dict[str, Any]]
    ) -> bool:

        # Extract frames and detections from the batch
        frames_batch = [item["frame"] for item in last_batch]

        detections_batch = [item["detections"] for item in last_batch]

        timestamp = last_batch[-1]["timestamp"]

        # Ensure we have enough frames
        if len(frames_batch) < self.window_size:
            return False

        # Ensure we have detections for all frames
        if any(d is None for d in detections_batch):
            return False

        # Preprocess each frame in the batch
        for frame, detections in zip(frames_batch, detections_batch):

            # Remove background and resize
            masked_frame = self.apply_background_mask(frame, detections)
            theft_frame = cv2.resize(masked_frame, (224, 224))

            # Convert to tensor and add to buffer
            t1 = time.time()
            tensor_frame = self.convert_frame_to_tensor(theft_frame)
            self.camera_tensor_buffers[camera_id].append(tensor_frame)
            t2 = time.time()
            logger.info(f"Time taken to convert frame to tensor: {t2 - t1:.4f}s")

        # Do theft inference if we have enough frames
        if len(self.camera_tensor_buffers[camera_id]) >= 100:
            # t1 = time.time()

            tensor_batch = list(self.camera_tensor_buffers[camera_id])

            # Send for prediction and get results
            should_trigger, theft_prob = self.theft_model.predict_with_tensors(
                camera_id=camera_id,
                tensors=tensor_batch,
            )
            t2 = time.time()
            logger.info(f"Time taken to predict: {t2 - t1:.4f}s")
            logger.info(f"Camera {camera_id} theft_prob={theft_prob:.4f}")

            # Verify if we should trigger an alert
            self.handle_theft_alert_trigger_logic(
                camera_id=camera_id,
                should_trigger=should_trigger,
                theft_prob=theft_prob,
                timestamp=timestamp,
            )

            # Clear the buffer after processing
            del tensor_batch, frames_batch, detections_batch

            return True

        else:
            return False

    def process_frame(self, data: Dict[str, Any]):
        """Process incoming frames data."""

        camera_id = data.get("camera_id", "unknown")

        state = self.get_state(camera_id)

        # We get a frame with detections per camera
        self.orig_frames_data[camera_id].append(data)

        # Increment frame count for the next batch
        state.frame_count += 1

        # Wait until we have enough frames for a batch
        if state.frame_count >= self.window_size:

            # Get the most recent window of frames
            recent_batch = list(self.orig_frames_data[camera_id])[-self.window_size :]

            # Process the recent batch for theft inference
            success = self.process_camera_batch(camera_id, recent_batch)

            # Reset frame count if we processed a batch
            if success:
                state.frame_count = 0

    def run(self):
        logger.info("Starting main loop...")

        # Setup poller for multiple ZMQ sockets
        poller = zmq.Poller()
        for receiver in self.receivers:
            poller.register(receiver, zmq.POLLIN)

        try:
            while True:
                try:
                    # Poll all receivers with timeout
                    socks = dict(poller.poll(timeout=5))
                    
                    # Process messages from any available receiver
                    for receiver in self.receivers:
                        if receiver in socks and socks[receiver] == zmq.POLLIN:
                            try:
                                data = receiver.recv_pyobj(zmq.NOBLOCK)
                                self.process_frame(data)
                            except zmq.Again:
                                pass
                            except Exception as e:
                                logger.error(f"Error receiving message: {e}")

                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        except Exception as e:
            logger.error(f"Critical error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        if hasattr(self, "api_service"):
            self.api_service.cleanup()
        
        if hasattr(self, "receivers"):
            for receiver in self.receivers:
                receiver.close()
                
        if hasattr(self, "video_sender"):
            self.video_sender.close()
            
        if hasattr(self, "context"):
            self.context.term()


def main():
    processor = TheftOrchestrator()
    processor.run()


if __name__ == "__main__":
    main()
