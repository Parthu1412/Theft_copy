#!/usr/bin/env python3
"""
Camera Orchestrator
- Manages camera threads and YOLO processing process
- Coordinates multiprocessing queue communication
- Handles process lifecycle and monitoring
"""

import os
import sys
import time
import signal
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from multiprocessing import Process, Queue, Event

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import config
from app.config import CameraConfig
from app.core.services.camera import CameraThreadService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("multiprocessing_orchestrator")


class ThreadInfo:
    """Information about a camera thread."""

    def __init__(self, thread: threading.Thread, camera_id: str) -> None:
        self.thread = thread
        self.camera_id = camera_id

    @classmethod
    def create(cls, thread: threading.Thread, camera_id: str) -> "ThreadInfo":
        """Create ThreadInfo from components."""
        return cls(thread=thread, camera_id=camera_id)




@dataclass
class CameraOrchestratorStats:
    """Statistics tracking for orchestrator performance."""

    start_time: float = field(default_factory=time.time)
    uptime: float = 0.0
    active_threads: int = 0
    total_threads: int = 0
    yolo_process_running: bool = False

    def update_stats(self, active_threads: int, total_threads: int, yolo_running: bool):
        """Update all statistics."""

        self.uptime = time.time() - self.start_time
        self.active_threads = active_threads
        self.total_threads = total_threads
        self.yolo_process_running = yolo_running

    def __str__(self) -> str:
        return (
            f"Uptime: {self.uptime:.1f}s, "
            f"Active threads: {self.active_threads}/{self.total_threads}, "
            f"YOLO process: {'Running' if self.yolo_process_running else 'Stopped'}"
        )


class CameraOrchestrator:
    def __init__(self):
        """Initialize the Camera Orchestrator"""

        self.camera_configs: Dict[int, CameraConfig] = {}
        self.frame_queue: Queue = Queue(maxsize=300)

        self.camera_threads: List[ThreadInfo] = []
        self.yolo_process: Optional[Process] = None
        self.stop_event = Event()

        # Statistics tracking
        self.stats = CameraOrchestratorStats()

        logger.info("Camera Orchestrator: Initialized")

    def load_camera_configs(self):
        """Load camera configurations from environment variables"""

        n = config.TOTAL_CAMERAS

        logger.info(f"Camera Orchestrator: Loading {n} camera configurations")

        for i in range(1, n + 1):
            self.camera_configs[i] = CameraConfig(
                id=os.getenv(f"CAMERA_ID_{i}", f"cam_{i}"),
                index=i,
                url=os.getenv(f"CAMERA_URL_{i}", ""),
                client_type=os.getenv(f"CLIENT_TYPE_{i}", "webrtc"),
                store_id=os.getenv(f"STORE_ID_{i}", "123"),
                websocket_url=os.getenv(f"WEBSOCKET_URL_{i}", ""),
            )

        logger.info(
            f"Camera Orchestrator: Loaded {len(self.camera_configs)} camera configurations"
        )
        return len(self.camera_configs) > 0

    def start_yolo_process(self):
        """Start YOLO processing process"""

        logger.info("Camera Orchestrator: Starting YOLO processing process...")

        # Create YOLO processor with the shared queue and stop event
        self.yolo_process = Process(
            target=self.run_yolo_processor,
            args=(self.frame_queue, self.stop_event),
            name="YOLO-Processor",
        )
        self.yolo_process.start()

        logger.info(
            f"Camera Orchestrator: YOLO process started with PID {self.yolo_process.pid}"
        )
        return True

    def run_yolo_processor(self, frame_queue, stop_event):
        """Run YOLO processor in separate process"""
        from app.core.services.yolo_inference import YoloInferenceService

        processor = YoloInferenceService(frame_queue, stop_event)
        processor.run()

    def start_camera_threads(self):
        """Start all camera threads"""

        for i, config in self.camera_configs.items():

            # Initialize CameraThread
            camera_thread = CameraThreadService(
                config=config,
                frame_queue=self.frame_queue,
                stop_event=self.stop_event,
            )

            # Create and start thread
            thread = threading.Thread(
                target=camera_thread.run, name=f"Camera-{config.id}"
            )
            thread.daemon = True

            # Create ThreadInfo and add to list
            thread_info = ThreadInfo.create(thread, config.id)
            self.camera_threads.append(thread_info)

            thread.start()

            logger.info(f"Camera Orchestrator: Started camera thread for {config.id}")

        logger.info("Camera Orchestrator: All camera threads started")
        return True

    def monitor_processes(self):
        """Monitor all processes and threads"""
        logger.info("Camera Orchestrator: Starting monitoring...")

        while not self.stop_event.is_set():
            try:
                # Check YOLO process
                if self.yolo_process and not self.yolo_process.is_alive():
                    logger.error("Camera Orchestrator: YOLO process died!")
                    break

                # Check camera threads

                if any(
                    not thread_info.thread.is_alive()
                    for thread_info in self.camera_threads
                ):
                    logger.warning(
                        "Camera Orchestrator: All camera threads have stopped"
                    )
                    break

                # Update statistics
                self.stats.update_stats(
                    active_threads=sum(
                        1 for t in self.camera_threads if t.thread.is_alive()
                    ),
                    total_threads=len(self.camera_threads),
                    yolo_running=(
                        self.yolo_process.is_alive() if self.yolo_process else False
                    ),
                )

                # Log queue statistics
                logger.info(
                    f"Camera Orchestrator: Queue size: {self.frame_queue.qsize()}"
                )

                # Periodic status update
                logger.info(f"Camera Orchestrator: {self.stats}")

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Camera Orchestrator: Error in monitoring: {e}")
                time.sleep(5)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Camera Orchestrator: Received signal {signum}, shutting down...")
        self.stop_event.set()

    def run(self):
        """Main run loop"""
        try:
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)

            # Load camera configurations
            if not self.load_camera_configs():
                logger.error(
                    "Camera Orchestrator: Failed to load camera configurations"
                )
                return

            # Start YOLO process
            if not self.start_yolo_process():
                logger.error("Camera Orchestrator: Failed to start YOLO process")
                return

            # Wait a moment for YOLO process to initialize
            time.sleep(3)

            # Start camera threads
            if not self.start_camera_threads():
                logger.error("Camera Orchestrator: Failed to start camera threads")
                return

            logger.info(
                "Camera Orchestrator: All systems running, starting monitoring..."
            )

            # Start monitoring
            self.monitor_processes()

        except KeyboardInterrupt:
            logger.info("Camera Orchestrator: Received interrupt signal")
        except Exception as e:
            logger.error(f"Camera Orchestrator: Critical error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Camera Orchestrator: Cleaning up...")

        # Signal all threads to stop
        self.stop_event.set()

        # Wait for all camera threads to finish
        logger.info("Camera Orchestrator: Waiting for camera threads to finish...")
        for thread_info in self.camera_threads:
            if thread_info.thread.is_alive():
                thread_info.thread.join(timeout=5.0)

        # Terminate YOLO process
        if self.yolo_process and self.yolo_process.is_alive():
            logger.info("Camera Orchestrator: Terminating YOLO process...")
            self.yolo_process.terminate()
            self.yolo_process.join(timeout=10.0)

            if self.yolo_process.is_alive():
                logger.warning("Camera Orchestrator: Force killing YOLO process...")
                self.yolo_process.kill()
                self.yolo_process.join()

        # Close multiprocessing queue
        self.frame_queue.close()
        self.frame_queue.join_thread()

        logger.info("Camera Orchestrator: Cleanup completed")


def main():
    """Main entry point"""
    logger.info("Camera Orchestrator: Starting...")

    orchestrator = CameraOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
