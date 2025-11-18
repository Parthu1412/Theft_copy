#!/usr/bin/env python3
"""
People Counting Orchestrator
- Receives frames and detections from YOLO processors via ZMQ
- Performs people counting and tracking
- Publishes counting data to Kafka
"""

import os
import sys
import time
import zmq
import logging
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import config
from app.people_counting.inference import PeopleCountingDetection
from app.kafka.asyncio.producer import CustomAIOKafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s'
)
logger = logging.getLogger("PeopleCountOrchestrator")


class PeopleCountOrchestrator:
    """
    Orchestrates people counting from multiple YOLO workers and cameras.
    """
    
    def __init__(self):
        """Initialize the people counting orchestrator"""
        self.context = zmq.Context()
        self.receivers = []
        
        # Create ZMQ receivers for each YOLO worker
        num_workers = config.NUM_YOLO_WORKERS
        for i in range(num_workers):
            receiver = self.context.socket(zmq.PULL)
            port = 5590 + i
            receiver.connect(f"tcp://localhost:{port}")
            self.receivers.append(receiver)
            logger.info(f"People Count Orchestrator: Connected to port {port} for YOLO worker {i}")
        
        # Kafka producer for sending counting data (initialize before loading regions)
        self.kafka_producer = None
        
        # Initialize people counting detectors per camera
        self.detectors: Dict[str, PeopleCountingDetection] = {}
        self._load_counting_regions()
        
        logger.info("People Count Orchestrator: Initialized successfully")
    
    def _load_counting_regions(self):
        """Load counting region configurations for cameras that have them"""
        try:
            for i in range(1, config.TOTAL_CAMERAS + 1):
                camera_id = os.getenv(f"CAMERA_ID_{i}")
                region_str = os.getenv(f"COUNTING_REGION_{i}")
                
                if camera_id and region_str:
                    # Parse region points (format: "x1-y1,x2-y2,x3-y3,x4-y4")
                    try:
                        points = []
                        for point_str in region_str.split(','):
                            x, y = map(int, point_str.split('-'))
                            points.append((x, y))
                        
                        moksa_camera_id = int(os.getenv(f"MOKSA_CAMERA_ID_{i}", "0"))
                        store_id = os.getenv(f"STORE_ID_{i}", config.STORE_ID)
                        
                        # Create detector for this camera
                        self.detectors[camera_id] = PeopleCountingDetection(
                            kafka_producer=self.kafka_producer,
                            region_points=points,
                            camera_id=camera_id,
                            camera_index=i,
                            moksa_camera_id=moksa_camera_id,
                            store_id=store_id,
                        )
                        logger.info(f"People Count Orchestrator: Loaded counting region for camera {camera_id}")
                    except Exception as e:
                        logger.error(f"People Count Orchestrator: Error parsing region for camera {i}: {e}")
                else:
                    logger.debug(f"People Count Orchestrator: No counting region for camera {i}")
                    
        except Exception as e:
            logger.error(f"People Count Orchestrator: Error loading counting regions: {e}")
    
    async def process_frame(self, data: Dict[str, Any]):
        """Process incoming frame data for people counting"""
        try:
            camera_id = data.get("camera_id")
            detections = data.get("detections", [])
            
            if camera_id not in self.detectors:
                # Camera doesn't have counting region configured
                return
            
            # Process detections through the detector
            await self.detectors[camera_id].detect(detections)
            
        except Exception as e:
            logger.error(f"People Count Orchestrator: Error processing frame: {e}")
    
    async def run_async(self):
        """Main async loop for receiving and processing frames"""
        logger.info("People Count Orchestrator: Starting async loop...")
        
        # Initialize Kafka producer
        try:
            self.kafka_producer = CustomAIOKafkaProducer()
            await self.kafka_producer.start()
            logger.info("People Count Orchestrator: Kafka producer started")
            
            # Update kafka producer in all detectors
            for detector in self.detectors.values():
                detector.kafka_producer = self.kafka_producer
                
        except Exception as e:
            logger.error(f"People Count Orchestrator: Failed to start Kafka producer: {e}")
        
        # Setup poller for multiple ZMQ sockets
        poller = zmq.Poller()
        for receiver in self.receivers:
            poller.register(receiver, zmq.POLLIN)
        
        try:
            while True:
                # Poll all receivers with timeout
                socks = dict(poller.poll(timeout=10))
                
                # Process messages from any available receiver
                for receiver in self.receivers:
                    if receiver in socks and socks[receiver] == zmq.POLLIN:
                        try:
                            data = receiver.recv_pyobj(zmq.NOBLOCK)
                            await self.process_frame(data)
                        except zmq.Again:
                            pass
                        except Exception as e:
                            logger.error(f"People Count Orchestrator: Error receiving message: {e}")
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("People Count Orchestrator: Received interrupt, shutting down...")
        except Exception as e:
            logger.error(f"People Count Orchestrator: Critical error in main loop: {e}")
        finally:
            await self.cleanup()
    
    def run(self):
        """Main entry point - runs async loop"""
        asyncio.run(self.run_async())
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("People Count Orchestrator: Cleaning up...")
        
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        for receiver in self.receivers:
            receiver.close()
        
        self.context.term()
        logger.info("People Count Orchestrator: Cleanup completed")


def main():
    """Main entry point"""
    logger.info("People Count Orchestrator: Starting...")
    orchestrator = PeopleCountOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
