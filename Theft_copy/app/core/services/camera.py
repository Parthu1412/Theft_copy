#!/usr/bin/env python3
"""
Camera Thread Module - I/O-bound frame reading
- Each camera runs in its own thread
- Reads frames from camera sources (RTSP, WebRTC, etc.)
- Puts frames into multiprocessing queue for YOLO processing
"""

import os
import cv2
import sys
import time
import queue
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventType

from app.utils.camera_intialize import CameraInit
from app.config import CameraConfig

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("camera_thread")


@dataclass
class CameraStats:
    """Statistics tracking for camera performance."""

    frame_number: int = 0
    frames_queued: int = 0

    def increment_read(self):
        """Increment frames read counter."""
        self.frame_number += 1

    def increment_queued(self):
        """Increment frames queued counter."""
        self.frames_queued += 1

    def reset(self):
        """Reset statistics counters."""
        self.frame_number = 0
        self.frames_queued = 0

    def __str__(self) -> str:
        return f"Read: {self.frame_number}, Queued: {self.frames_queued}"


class CameraThreadService:
    def __init__(
        self,
        config: CameraConfig,
        frame_queue: Queue,
        stop_event: EventType,
    ):
        """
        Initialize camera thread

        Args:
            camera_config: Camera configuration
            frame_queue: Multiprocessing queue to put frames into
            stop_event: Event to signal thread to stop
        """
        self.config = config
        self.frame_queue = frame_queue
        self.stop_event = stop_event

        # Statistics tracking
        self.stats = CameraStats()

        logger.info(
            f"Camera Thread {self.config.id}: Initialized for {self.config.url}"
        )

    def run(self):
        """Main camera thread loop"""
        logger.info(f"Camera Thread {self.config.id}: Starting...")

        try:
            # Initialize camera
            camera_init = CameraInit(self.config)
            cap = camera_init.camera_init()

            if cap is None:
                logger.error(
                    f"Camera Thread {self.config.id}: Failed to initialize camera"
                )
                return

            logger.info(f"Camera Thread {self.config.id}: Successfully opened camera")

        except ConnectionError as ce:
            logger.error(f"Camera Thread {self.config.id}: Connection error: {ce}")
            return

        try:

            while not self.stop_event.is_set():
                # Read frame
                ret, frame = cap.read()

                if not ret and isinstance(frame, np.ndarray):
                    continue

                if not ret or not isinstance(frame, np.ndarray):
                    logger.warning("Error loading frame")

                    cap.release()
                    cap = camera_init.camera_init()
                    continue

                # Update statistics
                self.stats.increment_read()
                timestamp = datetime.now().isoformat() + 'Z'

                # Resize frame before queuing (reduces memory and improves performance)
                resized_frame = cv2.resize(frame, (640, 480))

                # Create frame data
                frame_data = {
                    "camera_id": self.config.id,
                    "frame": resized_frame,
                    "frame_number": self.stats.frame_number,
                    "timestamp": timestamp,
                    "store_id": self.config.store_id,
                }

                # Try to put frame in queue (non-blocking)
                try:
                    self.frame_queue.put(frame_data, block=False)

                    self.stats.increment_queued()

                    # Log queue size periodically
                    if self.stats.frame_number % 100 == 0:
                        queue_size = self.frame_queue.qsize()
                        logger.info(
                            f"Camera Thread {self.config.id}: Queue size: {queue_size}, "
                            f"{self.stats}"
                        )
                        self.stats.reset()

                except queue.Full:
                    # Queue is full, drop frame
                    logger.warning(f"Camera Thread {self.config.id}: 1 dropped")

        except Exception as e:
            logger.error(f"Camera Thread {self.config.id}: Error: {e}")

        finally:
            # Release camera resource
            if cap:
                cap.release()
            logger.info(f"Camera Thread {self.config.id}: Stopped.")
