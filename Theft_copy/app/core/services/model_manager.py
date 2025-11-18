"""
Service Messaging Module - Handles ZMQ communication to different services
- Manages ZMQ sockets for theft detection, heatmap, and people counting
- Loads camera configurations for each service
- Routes processed frames to appropriate services
"""

import os
import time
import zmq
import logging
import numpy as np
from typing import List, Dict, Set

from app import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("Service Messenger")


class ModelManager:
    """Handles communication with different services via ZMQ."""
    
    def __init__(self, worker_id: int = 0):
        """Initialize ZMQ sockets and load camera configurations.
        
        Args:
            worker_id: ID of this YOLO worker (for multi-YOLO setups)
        """
        self.worker_id = worker_id
        self.context = zmq.Context()
        
        # Initialize theft detection
        self._setup_theft_detection()
        
        # Initialize heatmap (if enabled)
        self._setup_heatmap()
        
        # Initialize people counting (if enabled)
        self._setup_people_counting()
    
    def _setup_theft_detection(self):
        """Setup ZMQ socket and load cameras for theft detection."""
        self.theft_sender = self.context.socket(zmq.PUSH)
        port = 5590 + self.worker_id
        self.theft_sender.connect(f"tcp://localhost:{port}")
        logger.info(f"Service Messenger {self.worker_id}: Connected to theft orchestrator on port {port}")
        
        # Load cameras enabled for theft detection
        self.theft_enabled_cameras = set()
        self._load_theft_enabled_cameras()
    
    def _setup_heatmap(self):
        """Setup ZMQ socket and load cameras for heatmap service."""
        self.heatmap_sender = None
        self.heatmap_enabled_cameras = set()
        self.last_sent_time_1fps: Dict[str, float] = {}
        
        if config.ENABLE_HEATMAP:
            self.heatmap_sender = self.context.socket(zmq.PUSH)
            heatmap_port = 5509 + self.worker_id
            self.heatmap_sender.bind(f"tcp://*:{heatmap_port}")
            logger.info(f"Service Messenger {self.worker_id}: Bound heatmap sender to port {heatmap_port}")
            
            # Load cameras that have polygon configurations
            self._load_heatmap_enabled_cameras()
    
    def _setup_people_counting(self):
        """Setup ZMQ socket and load cameras for people counting service."""
        self.people_counting_sender = None
        self.people_counting_enabled_cameras = set()
        
        if config.ENABLE_PEOPLE_COUNTING:
            self.people_counting_sender = self.context.socket(zmq.PUSH)
            people_counting_port = 5590 + self.worker_id
            self.people_counting_sender.bind(f"tcp://*:{people_counting_port}")
            logger.info(f"Service Messenger {self.worker_id}: Bound people counting sender to port {people_counting_port}")
            
            # Load cameras that have counting region configurations
            self._load_people_counting_enabled_cameras()
    
    def _load_theft_enabled_cameras(self):
        """Load list of cameras enabled for theft detection."""
        enable_filtering = os.getenv("ENABLE_THEFT_FILTERING", "false").lower() == "true"
        
        if not enable_filtering:
            # If filtering disabled, enable all cameras for theft detection
            for i in range(1, config.TOTAL_CAMERAS + 1):
                camera_id = os.getenv(f"CAMERA_ID_{i}")
                if camera_id:
                    self.theft_enabled_cameras.add(camera_id)
            logger.info(
                f"Service Messenger {self.worker_id}: Theft filtering DISABLED - "
                f"all {len(self.theft_enabled_cameras)} cameras enabled for theft detection"
            )
        else:
            # If filtering enabled, only use cameras with ENABLE_THEFT_{i}=true
            for i in range(1, config.TOTAL_CAMERAS + 1):
                camera_id = os.getenv(f"CAMERA_ID_{i}")
                enable_theft = os.getenv(f"ENABLE_THEFT_{i}", "true").lower() == "true"
                
                if camera_id and enable_theft:
                    self.theft_enabled_cameras.add(camera_id)
                    logger.info(f"Service Messenger {self.worker_id}: Camera {camera_id} enabled for theft detection")
            
            logger.info(
                f"Service Messenger {self.worker_id}: Theft filtering ENABLED - "
                f"{len(self.theft_enabled_cameras)} cameras enabled for theft detection: "
                f"{self.theft_enabled_cameras}"
            )
    
    def _load_heatmap_enabled_cameras(self):
        """Load list of cameras that have polygon configurations for heatmap."""
        for i in range(1, config.TOTAL_CAMERAS + 1):
            camera_id = os.getenv(f"CAMERA_ID_{i}")
            polygon_str = os.getenv(f"POLYGON_{i}")
            polygon_info_str = os.getenv(f"POLYGON_{i}")
            
            # Only cameras with both polygon and polygon_info are heatmap-enabled
            if camera_id and polygon_str and polygon_info_str:
                self.heatmap_enabled_cameras.add(camera_id)
                logger.info(f"Service Messenger {self.worker_id}: Camera {camera_id} enabled for heatmap")
        
        logger.info(
            f"Service Messenger {self.worker_id}: {len(self.heatmap_enabled_cameras)} cameras enabled for heatmap: "
            f"{self.heatmap_enabled_cameras}"
        )
    
    def _load_people_counting_enabled_cameras(self):
        """Load list of cameras that have counting region configurations for people counting."""
        for i in range(1, config.TOTAL_CAMERAS + 1):
            camera_id = os.getenv(f"CAMERA_ID_{i}")
            counting_region_str = os.getenv(f"COUNTING_REGION_{i}")
            
            # Only cameras with counting region are enabled
            if camera_id and counting_region_str:
                self.people_counting_enabled_cameras.add(camera_id)
                logger.info(f"Service Messenger {self.worker_id}: Camera {camera_id} enabled for people counting")
        
        logger.info(
            f"Service Messenger {self.worker_id}: {len(self.people_counting_enabled_cameras)} cameras enabled for people counting: "
            f"{self.people_counting_enabled_cameras}"
        )
    
    def send_to_theft_detection(
        self,
        camera_id: str,
        frame_number: int,
        frame: np.ndarray,
        timestamp: float,
        detections: List[List[int]],
        store_id: str,
    ):
        """Sends processed frame data to theft detection via ZMQ.
        
        Args:
            camera_id: The ID of the camera
            frame_number: The frame number
            frame: The processed frame
            timestamp: The timestamp of the frame
            detections: The YOLO detections
            store_id: The store ID associated with the camera
        """
        # Check if camera is enabled for theft detection
        if camera_id not in self.theft_enabled_cameras:
            return
        
        message = {
            "camera_id": camera_id,
            "frame": frame,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "detections": detections,
            "store_id": store_id,
        }
        
        try:
            self.theft_sender.send_pyobj(message, zmq.NOBLOCK)
            logger.info(
                f"Service Messenger {self.worker_id}: Sent frame {frame_number} from camera {camera_id} "
                f"with {len(detections)} detections to theft detection"
            )
        except zmq.Again:
            logger.warning(
                f"Service Messenger {self.worker_id}: Theft ZMQ queue full, "
                f"dropping frame for {camera_id} (frame {frame_number})"
            )
        except Exception as e:
            logger.error(f"Service Messenger {self.worker_id}: Failed to send to theft detection: {e}")
    
    def send_to_heatmap(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: List[List[int]],
    ):
        """Sends frame and detections to heatmap service via ZMQ (at 1 FPS).
        
        Args:
            camera_id: The ID of the camera
            frame: The processed frame
            detections: The YOLO detections
        """
        if not self.heatmap_sender or camera_id not in self.heatmap_enabled_cameras:
            return
        
        current_time = time.time()
        last_time = self.last_sent_time_1fps.get(camera_id, 0)
        
        # Send at 1 FPS (one frame per second)
        if current_time - last_time >= 1.0:
            message = {
                "camera_id": camera_id,
                "frame": frame,
                "detections": detections,
                "timestamp": current_time,
            }
            
            try:
                self.heatmap_sender.send_pyobj(message, zmq.NOBLOCK)
                self.last_sent_time_1fps[camera_id] = current_time
                logger.info(f"Service Messenger {self.worker_id}: Sent 1 FPS frame to heatmap for camera {camera_id}")
            except zmq.Again:
                logger.warning(
                    f"Service Messenger {self.worker_id}: Heatmap ZMQ queue full, "
                    f"dropping frame for {camera_id}"
                )
            except Exception as e:
                logger.error(f"Service Messenger {self.worker_id}: Failed to send to heatmap: {e}")
    
    def send_to_people_counting(
        self,
        camera_id: str,
        frame_number: int,
        frame: np.ndarray,
        timestamp: float,
        detections: List[List[int]],
        store_id: str
    ):
        """Sends processed frame data to people counting service via ZMQ.
        
        Args:
            camera_id: Camera identifier
            frame_number: Frame number
            frame: Frame image
            timestamp: Timestamp
            detections: YOLO detections in format [[x1, y1, x2, y2], ...]
            store_id: Store identifier
        """
        if not self.people_counting_sender or camera_id not in self.people_counting_enabled_cameras:
            return
        
        # Format detections for Redis_Detection class (expects dicts with 'box' key)
        formatted_detections = [{"box": det[:4]} for det in detections if len(det) >= 4]
        
        message = {
            "camera_id": camera_id,
            "detections": formatted_detections,
            "timestamp": timestamp,
        }
        
        try:
            self.people_counting_sender.send_pyobj(message, zmq.NOBLOCK)
            logger.info(
                f"Service Messenger {self.worker_id}: Sent frame {frame_number} from camera {camera_id} "
                f"with {len(formatted_detections)} detections to people counting"
            )
        except zmq.Again:
            logger.warning(
                f"Service Messenger {self.worker_id}: People counting ZMQ queue full, "
                f"dropping frame for {camera_id} (frame {frame_number})"
            )
        except Exception as e:
            logger.error(f"Service Messenger {self.worker_id}: Failed to send to people counting: {e}")
    
    def cleanup(self):
        """Cleans up ZMQ resources."""
        if self.theft_sender:
            self.theft_sender.close()
        if self.heatmap_sender:
            self.heatmap_sender.close()
        if self.people_counting_sender:
            self.people_counting_sender.close()
        self.context.term()
        logger.info(f"Service Messenger {self.worker_id}: Cleaned up ZMQ resources.")

