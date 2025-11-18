#!/usr/bin/env python3
"""
Camera Orchestrator with Multi-YOLO Support
- Manages camera threads and multiple YOLO processing processes
- Coordinates multiprocessing queue communication
- Handles process lifecycle and monitoring
- Distributes cameras across YOLO workers in round-robin fashion
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
logger = logging.getLogger("multi_yolo_orchestrator")


class ThreadInfo:
    """Information about a camera thread."""

    def __init__(self, thread: threading.Thread, camera_id: str, worker_id: int) -> None:
        self.thread = thread
        self.camera_id = camera_id
        self.worker_id = worker_id

    @classmethod
    def create(cls, thread: threading.Thread, camera_id: str, worker_id: int) -> "ThreadInfo":
        """Create ThreadInfo from components."""
        return cls(thread=thread, camera_id=camera_id, worker_id=worker_id)


@dataclass
class CameraOrchestratorStats:
    """Statistics tracking for orchestrator performance."""

    start_time: float = field(default_factory=time.time)
    uptime: float = 0.0
    active_threads: int = 0
    total_threads: int = 0
    yolo_processes_running: int = 0

    def update_stats(self, active_threads: int, total_threads: int, yolo_running: int):
        """Update all statistics."""

        self.uptime = time.time() - self.start_time
        self.active_threads = active_threads
        self.total_threads = total_threads
        self.yolo_processes_running = yolo_running

    def __str__(self) -> str:
        return (
            f"Uptime: {self.uptime:.1f}s, "
            f"Active threads: {self.active_threads}/{self.total_threads}, "
            f"YOLO processes: {self.yolo_processes_running} running"
        )


class MultiYOLOCameraOrchestrator:
    def __init__(self):
        """Initialize the Multi-YOLO Camera Orchestrator"""

        self.camera_configs: Dict[int, CameraConfig] = {}
        self.num_yolo_workers = config.NUM_YOLO_WORKERS
        
        # Create multiple frame queues (one per YOLO worker)
        self.frame_queues: List[Queue] = []
        for i in range(self.num_yolo_workers):
            self.frame_queues.append(Queue(maxsize=3000))

        self.camera_threads: List[ThreadInfo] = []
        self.yolo_processes: List[Process] = []
        self.stop_event = Event()

        # Statistics tracking
        self.stats = CameraOrchestratorStats()

        logger.info(f"Multi-YOLO Camera Orchestrator: Initialized with {self.num_yolo_workers} YOLO workers")

    def load_camera_configs(self):
        """Load camera configurations from environment variables"""

        n = config.TOTAL_CAMERAS

        logger.info(f"Multi-YOLO Camera Orchestrator: Loading {n} camera configurations")

        for i in range(1, n + 1):
            self.camera_configs[i] = CameraConfig(
                id=os.getenv(f"CAMERA_ID_{i}", f"cam_{i}"),
                index=i,
                url=os.getenv(f"CAMERA_URL_{i}", ""),
                client_type=os.getenv(f"CLIENT_TYPE_{i}", "webrtc"),
                store_id=os.getenv(f"STORE_ID_{i}", config.STORE_ID),
                websocket_url=os.getenv(f"WEBSOCKET_URL_{i}", ""),
                moksa_camera_id=int(os.getenv(f"MOKSA_CAMERA_ID_{i}", "0")),
            )

        # time.sleep(10)

        logger.info(
            f"Multi-YOLO Camera Orchestrator: Loaded {len(self.camera_configs)} camera configurations"
        )
        
        # Log camera distribution across YOLO workers
        for worker_id in range(self.num_yolo_workers):
            cameras = [
                cfg.id for idx, cfg in self.camera_configs.items() 
                if (idx - 1) % self.num_yolo_workers == worker_id
            ]
            logger.info(f"YOLO Worker {worker_id}: Assigned cameras {cameras}")
        
        return len(self.camera_configs) > 0

    def start_yolo_processes(self):
        """Start multiple YOLO processing processes"""

        logger.info(f"Multi-YOLO Camera Orchestrator: Starting {self.num_yolo_workers} YOLO processing processes...")

        for worker_id in range(self.num_yolo_workers):
            # Create YOLO processor with dedicated queue and stop event
            yolo_process = Process(
                target=self.run_yolo_processor,
                args=(self.frame_queues[worker_id], self.stop_event, worker_id),
                name=f"YOLO-Processor-{worker_id}",
            )
            yolo_process.start()
            self.yolo_processes.append(yolo_process)

            logger.info(
                f"Multi-YOLO Camera Orchestrator: YOLO process {worker_id} started with PID {yolo_process.pid}"
            )
        
        return True

    def run_yolo_processor(self, frame_queue, stop_event, worker_id):
        """Run YOLO processor in separate process"""
        from app.core.services.yolo_inference import YoloInferenceService

        logger.info(f"YOLO Worker {worker_id}: Starting processor")
        processor = YoloInferenceService(frame_queue, stop_event, worker_id=worker_id)
        processor.run()

    def start_camera_threads(self):
        """Start all camera threads with round-robin YOLO worker assignment"""

        for i, camera_config in self.camera_configs.items():
            # Assign camera to YOLO worker using round-robin
            worker_id = (i - 1) % self.num_yolo_workers
            assigned_queue = self.frame_queues[worker_id]

            # Initialize CameraThread
            camera_thread = CameraThreadService(
                config=camera_config,
                frame_queue=assigned_queue,
                stop_event=self.stop_event,
            )

            # Create and start thread
            thread = threading.Thread(
                target=camera_thread.run, name=f"Camera-{camera_config.id}"
            )
            thread.daemon = True

            # Create ThreadInfo and add to list
            thread_info = ThreadInfo.create(thread, camera_config.id, worker_id)
            self.camera_threads.append(thread_info)

            thread.start()

            logger.info(
                f"Multi-YOLO Camera Orchestrator: Started camera thread for {camera_config.id} "
                f"(assigned to YOLO Worker {worker_id})"
            )

        logger.info("Multi-YOLO Camera Orchestrator: All camera threads started")
        return True

    def monitor_processes(self):
        """Monitor all processes and threads"""
        logger.info("Multi-YOLO Camera Orchestrator: Starting monitoring...")

        while not self.stop_event.is_set():
            try:
                # Check YOLO processes
                alive_yolo_processes = sum(1 for p in self.yolo_processes if p.is_alive())
                
                if alive_yolo_processes < self.num_yolo_workers:
                    logger.error(
                        f"Multi-YOLO Camera Orchestrator: Some YOLO processes died! "
                        f"({alive_yolo_processes}/{self.num_yolo_workers} alive)"
                    )
                    break

                # Check camera threads
                if any(
                    not thread_info.thread.is_alive()
                    for thread_info in self.camera_threads
                ):
                    logger.warning(
                        "Multi-YOLO Camera Orchestrator: Some camera threads have stopped"
                    )

                # Update statistics
                self.stats.update_stats(
                    active_threads=sum(
                        1 for t in self.camera_threads if t.thread.is_alive()
                    ),
                    total_threads=len(self.camera_threads),
                    yolo_running=alive_yolo_processes,
                )

                # Log queue statistics for each worker
                for worker_id, queue in enumerate(self.frame_queues):
                    logger.info(
                        f"Multi-YOLO Camera Orchestrator: Worker {worker_id} queue size: {queue.qsize()}"
                    )

                # Periodic status update
                logger.info(f"Multi-YOLO Camera Orchestrator: {self.stats}")

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Multi-YOLO Camera Orchestrator: Error in monitoring: {e}")
                time.sleep(5)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Multi-YOLO Camera Orchestrator: Received signal {signum}, shutting down...")
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
                    "Multi-YOLO Camera Orchestrator: Failed to load camera configurations"
                )
                return

            # Start YOLO processes
            if not self.start_yolo_processes():
                logger.error("Multi-YOLO Camera Orchestrator: Failed to start YOLO processes")
                return

            # Wait a moment for YOLO processes to initialize
            time.sleep(3)

            # Start camera threads
            if not self.start_camera_threads():
                logger.error("Multi-YOLO Camera Orchestrator: Failed to start camera threads")
                return

            logger.info(
                "Multi-YOLO Camera Orchestrator: All systems running, starting monitoring..."
            )

            # Start monitoring
            self.monitor_processes()

        except KeyboardInterrupt:
            logger.info("Multi-YOLO Camera Orchestrator: Received interrupt signal")
        except Exception as e:
            logger.error(f"Multi-YOLO Camera Orchestrator: Critical error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Multi-YOLO Camera Orchestrator: Cleaning up...")

        # Signal all threads to stop
        self.stop_event.set()

        # Wait for all camera threads to finish
        logger.info("Multi-YOLO Camera Orchestrator: Waiting for camera threads to finish...")
        for thread_info in self.camera_threads:
            if thread_info.thread.is_alive():
                thread_info.thread.join(timeout=5.0)

        # Terminate YOLO processes
        logger.info("Multi-YOLO Camera Orchestrator: Terminating YOLO processes...")
        for i, yolo_process in enumerate(self.yolo_processes):
            if yolo_process.is_alive():
                logger.info(f"Multi-YOLO Camera Orchestrator: Terminating YOLO process {i}...")
                yolo_process.terminate()
                yolo_process.join(timeout=10.0)

                if yolo_process.is_alive():
                    logger.warning(f"Multi-YOLO Camera Orchestrator: Force killing YOLO process {i}...")
                    yolo_process.kill()
                    yolo_process.join()

        # Close multiprocessing queues
        for i, queue in enumerate(self.frame_queues):
            logger.info(f"Multi-YOLO Camera Orchestrator: Closing queue {i}...")
            queue.close()
            queue.join_thread()

        logger.info("Multi-YOLO Camera Orchestrator: Cleanup completed")


def main():
    """Main entry point"""
    logger.info("Multi-YOLO Camera Orchestrator: Starting...")

    orchestrator = MultiYOLOCameraOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()

