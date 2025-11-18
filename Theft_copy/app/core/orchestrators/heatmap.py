#!/usr/bin/env python3
"""
Heatmap Orchestrator
- Receives frames and detections from YOLO processors via ZMQ
- Generates heatmaps for customer movement tracking
- Processes aisle statistics
- Publishes heatmap data at configured intervals
"""

import os
import sys
import time
import zmq
import logging
import numpy as np
import datetime
import asyncio
from collections import defaultdict
from typing import Dict, List, Any
from queue import Queue
import threading
from zoneinfo import ZoneInfo
from app import tzinfo

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import config
from app.heatmap.heat_aisle import AisleTracker
from app.heatmap.processing import ProcessingPolygons
from app.heatmap.postprocess import Post_process
from app.kafka.asyncio.producer import CustomAIOKafkaProducer
from app.utils.message import HeatMapMessage, AisleMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s'
)
logger = logging.getLogger("HeatmapOrchestrator")


class HeatmapOrchestrator:
    """
    Orchestrates heatmap generation from multiple YOLO workers and cameras.
    """

    def __init__(self):
        """Initialize the heatmap orchestrator"""
        self.context = zmq.Context()
        self.receivers = []
        
        # Create ZMQ receivers for each YOLO worker
        num_workers = config.NUM_YOLO_WORKERS
        for i in range(num_workers):
            receiver = self.context.socket(zmq.PULL)
            port = 5509 + i
            receiver.connect(f"tcp://localhost:{port}")
            self.receivers.append(receiver)
            logger.info(f"Heatmap Orchestrator: Connected to port {port} for YOLO worker {i}")
        
        # Initialize polygon processor for aisle tracking
        self.polygon_processor = ProcessingPolygons()
        self.load_polygon_configs()
        
        # Initialize aisle trackers per camera
        self.aisle_trackers: Dict[str, AisleTracker] = {}
        self.initialize_aisle_trackers()
        
        # Heatmaps storage
        self.camera_data: Dict[str, Dict[str, Any]] = {}
        
        # Queue for postprocessing
        self.postprocess_queue = Queue()
        self.postprocessor = Post_process(self.postprocess_queue)
        
        # Start queue processor thread
        self.queue_processor_thread = threading.Thread(
            target=self.process_queue, 
            daemon=True
        )
        self.queue_processor_thread.start()
        
        # Timing
        self.start_time = time.time()
        self.last_reset_time = time.time()
        
        logger.info("Heatmap Orchestrator: Initialized successfully")

    def load_polygon_configs(self):
        """Load polygon configurations for all cameras"""
        try:
            # Load polygons from environment variables
            for i in range(1, config.TOTAL_CAMERAS + 1):
                camera_id = os.getenv(f"CAMERA_ID_{i}")
                polygon_str = os.getenv(f"POLYGON_{i}")
                polygon_info_str = os.getenv(f"POLYGON_INFO_{i}")
                
                if camera_id and polygon_str and polygon_info_str:
                    self.polygon_processor.parse_camera_polygons(
                        camera_id, polygon_str, polygon_info_str
                    )
                    logger.info(f"Heatmap Orchestrator: Loaded polygons for camera {camera_id}")
                else:
                    logger.warning(f"Heatmap Orchestrator: Missing polygon config for camera {i}")
                    
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Error loading polygon configs: {e}")

    def initialize_aisle_trackers(self):
        """Initialize aisle trackers for each camera with polygon data"""
        try:
            for i in range(1, config.TOTAL_CAMERAS + 1):
                camera_id = os.getenv(f"CAMERA_ID_{i}")
                store_id = os.getenv(f"STORE_ID_{i}", config.STORE_ID)
                
                if camera_id:
                    polygons = self.polygon_processor.get_camera_polygons(camera_id)
                    
                    if polygons:
                        tracker = AisleTracker(polygons, store_id, camera_id)
                        self.aisle_trackers[camera_id] = tracker
                        logger.info(f"Heatmap Orchestrator: Initialized tracker for camera {camera_id}")
                    else:
                        logger.warning(f"Heatmap Orchestrator: No polygons found for camera {camera_id}")
                        
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Error initializing trackers: {e}")

    def process_frame(self, camera_id: str, frame: np.ndarray, detections: List[List[int]]):
        """
        Process a frame with detections for heatmap generation.
        
        Args:
            camera_id: Camera identifier
            frame: Frame image
            detections: List of bounding box detections [x1, y1, x2, y2]
        """
        try:
            tracker = self.aisle_trackers.get(camera_id)
            
            if tracker is None:
                logger.warning(f"Heatmap Orchestrator: No tracker found for camera {camera_id}")
                return
            
            # Process frame and get heatmap + aisle info
            heatmap, aisle_stats = tracker.detect(frame, detections)
            
            # Store results (will be aggregated periodically)
            logger.debug(f"Heatmap Orchestrator: Processed frame for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Error processing frame for camera {camera_id}: {e}")

    def aggregate_data(self):
        """Aggregate heatmaps and aisle info from all cameras (per-camera basis)"""
        try:
            # Store camera-specific data instead of combining
            self.camera_data = {}
            
            for camera_id, tracker in self.aisle_trackers.items():
                # Get heatmap for this camera
                if tracker.heatmap is not None:
                    heatmap = tracker.heatmap.copy()
                else:
                    heatmap = np.zeros((480, 640), dtype=np.float32)
                
                # Get aisle statistics for this camera
                stats = tracker.get_statistics()
                
                # Extract aisle info (stats contain camera id as key)
                aisle_info = {}
                for camera_id, cam_data in stats.items():
                    if not isinstance(cam_data, dict):
                        logger.warning(f"Skipping malformed data for camera {camera_id}: {cam_data}")

                aisle_info[camera_id] = cam_data
                
                
                
                # Store per-camera data
                self.camera_data[camera_id] = {
                    "heatmap": heatmap,
                    "aisle_info": aisle_info,
                    "store_id": tracker.store_id
                }
            
            logger.info(f"Heatmap Orchestrator: Aggregated data for {len(self.camera_data)} cameras")
            
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Error aggregating data: {e}")

    def reset_all_trackers(self):
        """Reset all trackers' heatmaps and statistics"""
        try:
            for camera_id, tracker in self.aisle_trackers.items():
                tracker.reset_heatmap()
                tracker.reset_statistics()
            
            logger.info("Heatmap Orchestrator: Reset all trackers")
            
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Error resetting trackers: {e}")

    def process_queue(self):
        """Process items from the postprocessing queue (runs in separate thread)"""
        logger.info("Heatmap Orchestrator: Queue processor thread started")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            try:
                # Get camera-specific data from queue (already compressed)
                payload = self.postprocess_queue.get(timeout=1.0)
                
                if not isinstance(payload, dict):
                    logger.warning("Heatmap Orchestrator: Invalid queue data structure. Expected dict.")
                    continue
                
                logger.info(f"Heatmap Orchestrator: Received heatmap payload for {len(payload)} cameras")
                
                # Send to Kafka (per-camera messages, heatmap already compressed)
                loop.run_until_complete(self.send_to_kafka_per_camera(payload))
                
            except Exception as e:
                if "Empty" not in str(e):
                    logger.error(f"Heatmap Orchestrator: Error in queue processor: {e}")
                time.sleep(0.1)

    async def send_to_kafka_per_camera(self, payload: Dict[str, Any]):
        """Send individual heatmap and aisle info messages per camera to Kafka"""
        try:
            kafka_producer = CustomAIOKafkaProducer()
            await kafka_producer.start()
            
            
            # Get moksa camera ID to store ID mapping
            moksa_to_store = {}
            for i in range(1, config.TOTAL_CAMERAS + 1):
                camera_id = os.getenv(f"CAMERA_ID_{i}", f"cam_{i:03d}")
                store_id = os.getenv(f"STORE_ID_{i}", config.STORE_ID)
                moksa_to_store[camera_id] = {
                    "moksa_id": camera_id,
                    "store_id": store_id
                }
            
            # Process each camera individually
            for camera_id_str, camera_data in payload.items():
                try:
                    heatmap_compressed = camera_data.get("heatmap")  # Already compressed string!
                    aisle_info = camera_data.get("aisle_info", {})
                    
                    if heatmap_compressed is None:
                        logger.warning(f"Heatmap Orchestrator: No heatmap for camera '{camera_id_str}'. Skipping.")
                        continue
                    
                    # Get moksa camera ID and store ID from mapping
                    mapping = moksa_to_store.get(camera_id_str, {})
                    moksa_camera_id = mapping.get("moksa_id", camera_id_str)
                    store_id = mapping.get("store_id", config.STORE_ID)
                    
                    # Build HeatMap Kafka message for THIS camera (heatmap already compressed)
                    heatmap_message = HeatMapMessage(
                        store_id=int(store_id),
                        moksa_camera_id=int(moksa_camera_id) if str(moksa_camera_id).isdigit() else 0,
                        heat_map=heatmap_compressed,  # Already compressed string!
                        timestamp=datetime.datetime.now(tz=tzinfo),
                        model_version="v3.0.0"
                    )
                    
                    # Send heatmap to Kafka
                    logger.info(f"Heatmap Orchestrator: [Store {store_id} - Camera {camera_id_str}] Heatmap message ready")
                    await kafka_producer.produce(
                        topic=config.KAFKA_HEAT_MAP_TOPIC,
                        message=heatmap_message
                    )
                    logger.info(f"Heatmap Orchestrator: [Store {store_id} - Camera {camera_id_str}] Heatmap sent to Kafka")
                    logger.debug(f"Heatmap message: {heatmap_message.to_dict()}")
                    
                    # Send aisle info if enabled and available
                    if config.AISLE_INFO_ENABLED and aisle_info:
                        aisle_message = AisleMessage(
                            store_id=int(store_id),
                            moksa_camera_id=int(moksa_camera_id) if str(moksa_camera_id).isdigit() else 0,
                            aisle_info=aisle_info[moksa_camera_id],
                            timestamp=datetime.datetime.now(tz=tzinfo),
                            model_version="v3.0.0"
                        )
                        
                        logger.info(f"Heatmap Orchestrator: [Store {store_id} - Camera {camera_id_str}] Aisle message ready")
                        await kafka_producer.produce(
                            topic=config.KAFKA_AISLE_INFO_TOPIC,
                            message=aisle_message
                        )
                        logger.info(f"Heatmap Orchestrator: [Store {store_id} - Camera {camera_id_str}] Aisle info sent to Kafka")
                        logger.debug(f"Aisle info message: {aisle_message.to_dict()}")
                
                except Exception as camera_error:
                    logger.error(f"Heatmap Orchestrator: Error processing camera {camera_id_str}: {camera_error}")
                    continue
            
            await kafka_producer.stop()
            logger.info(f"Heatmap Orchestrator: Successfully sent messages for {len(payload)} cameras to Kafka")
            
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Error sending to Kafka: {e}", exc_info=True)

    def run(self):
        """Main run loop - receives frames from YOLO workers and processes heatmaps"""
        logger.info("Heatmap Orchestrator: Starting main loop...")
        
        # Setup poller for multiple ZMQ sockets
        poller = zmq.Poller()
        for receiver in self.receivers:
            poller.register(receiver, zmq.POLLIN)
        
        try:
            while True:
                # Poll all receivers with timeout
                socks = dict(poller.poll(timeout=100))
                
                # Process messages from any available receiver
                for receiver in self.receivers:
                    if receiver in socks and socks[receiver] == zmq.POLLIN:
                        try:
                            # Receive message
                            message = receiver.recv_pyobj(zmq.NOBLOCK)
                            
                            camera_id = message.get("camera_id")
                            frame = message.get("frame")
                            detections = message.get("detections", [])
                            
                            if camera_id and frame is not None:
                                self.process_frame(camera_id, frame, detections)
                            
                        except zmq.Again:
                            pass
                        except Exception as e:
                            logger.error(f"Heatmap Orchestrator: Error receiving message: {e}")
                
                # Check if it's time to aggregate and reset
                current_time = time.time()
                if current_time - self.last_reset_time >= config.HEAT_INTERVAL:
                    logger.info("Heatmap Orchestrator: Heat interval reached, aggregating data...")
                    
                    # Aggregate per-camera data
                    self.aggregate_data()
                    
                    # Postprocess and push to queue
                    self.postprocessor.image_postprocess(self.camera_data)
                    self.postprocessor.push_to_queue()
                    
                    # Reset trackers
                    self.reset_all_trackers()
                    
                    self.last_reset_time = current_time
                    logger.info("Heatmap Orchestrator: Data aggregated and trackers reset")
                
        except KeyboardInterrupt:
            logger.info("Heatmap Orchestrator: Received interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Heatmap Orchestrator: Critical error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Heatmap Orchestrator: Cleaning up...")
        
        for receiver in self.receivers:
            receiver.close()
        
        self.context.term()
        logger.info("Heatmap Orchestrator: Cleanup completed")


def main():
    """Main entry point"""
    if not config.ENABLE_HEATMAP:
        logger.warning("Heatmap Orchestrator: ENABLE_HEATMAP is false, exiting...")
        return
    
    logger.info("Heatmap Orchestrator: Starting...")
    orchestrator = HeatmapOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
