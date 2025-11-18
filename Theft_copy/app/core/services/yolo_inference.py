
"""
YOLO Processing Module - CPU-bound processing
- Runs in separate process
- Pulls frames from multiprocessing queue
- Runs YOLO inference and interpolation
- Sends results to theft detection via ZMQ
"""

import os
import sys
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any, List, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing import Queue

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from app import config
from app.core.data_models.latency import LatencyStats
from app.core.services.model_manager import ModelManager

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("YOLO Processor")


@dataclass
class CameraState:
    frames_since_last_yolo: int = 0
    last_detections: List[List[int]] = field(default_factory=list)
    current_detections: List[List[int]] = field(default_factory=list)

    def update_state(self, detections: List[List[int]]):
        self.frames_since_last_yolo = 0
        self.last_detections = self.current_detections
        self.current_detections = detections


class YoloInferenceService:
    """Processes frames using YOLO model and sends results via ZMQ."""

    def __init__(self, frame_queue: "Queue[Dict[str, Any]]", stop_event=None, worker_id: int = 0):
        """Initializes the YoloInferenceService with a frame queue.

        Args:
            frame_queue (Queue): Multiprocessing queue to receive frames.
            stop_event: Event to signal when to stop processing.
            worker_id (int): ID of this YOLO worker (for multi-YOLO setups).
        """
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.worker_id = worker_id

        # Initialize service messenger for ZMQ communication
        self.model_manager = ModelManager(worker_id=worker_id)
        
        # YOLO inference latency tracking
        self.inf_yolo_latency = []

        # Camera states for interpolation
        self.camera_states: Dict[str, CameraState] = {}

        # YOLO latency stats
        self.latency_stats = LatencyStats()

    def get_camera_state(self, camera_id: str) -> CameraState:
        """Retrieve or initialize the state for a given camera."""

        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = CameraState()

        return self.camera_states[camera_id]

    def load_yolo_model(self):
        """Loads the YOLO model."""

        logger.info("YOLO Processor: Loading YOLO model...")

        model_path = config.YOLO_MODEL_PATH

        try:
            self.yolo_model = YOLO(model_path).to("cuda:0")
            logger.info(f"YOLO Processor: Loaded YOLO model from {model_path}")

            # Warmup the model
            dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)

            _ = self.yolo_model(dummy_frame, verbose=False)

            logger.info("YOLO Processor: YOLO model warmup completed")

        except Exception as e:
            logger.error(f"YOLO Processor: Failed to load YOLO model: {e}")

            raise

    def run_yolo_inference(self, frame: np.ndarray) -> List[List[int]]:
        """Runs YOLO inference on a given frame.

        Args:
            frame (np.ndarray): The input frame for inference.

        Returns:
            list: The results from the YOLO model.
        """
        results = self.yolo_model(
            frame, verbose=True, classes=[1], conf=config.YOLO_CONFIDENCE_THRESHOLD
        )

        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy
            detections = boxes.int().tolist()

        return detections

    def iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two boxes [x1,y1,x2,y2]."""

        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / float(area1 + area2 - inter)

    def match_detections(
        self,
        last_detections: List[List[int]],
        next_detections: List[List[int]],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Return matched pairs, unmatched_last, unmatched_next."""

        if not last_detections or not next_detections:
            return ([], list(range(len(last_detections))))

        # Build cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(last_detections), len(next_detections)))
        for i, b1 in enumerate(last_detections):
            for j, b2 in enumerate(next_detections):
                cost_matrix[i, j] = 1 - self.iou(b1, b2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = list(zip(row_ind, col_ind))
        unmatched_last = [i for i in range(len(last_detections)) if i not in row_ind]

        return matched_pairs, unmatched_last

    def interpolate_boxes(
        self, box1: List[int], box2: List[int], alpha: float
    ) -> List[int]:
        """Interpolate between two boxes."""

        return [
            int(box1[0] * (1 - alpha) + box2[0] * alpha),
            int(box1[1] * (1 - alpha) + box2[1] * alpha),
            int(box1[2] * (1 - alpha) + box2[2] * alpha),
            int(box1[3] * (1 - alpha) + box2[3] * alpha),
        ]

    def interpolate_detections(
        self,
        last_detections: List[List[int]],
        current_detections: List[List[int]],
        frames_in_between: int,
    ) -> List[List[int]]:
        """Interpolate detections between two frames."""

        if frames_in_between <= 0:
            return last_detections

        matched_pairs, unmatched_last = self.match_detections(
            last_detections, current_detections
        )

        alpha = frames_in_between / config.YOLO_FRAMES_TO_INTERPOLATE

        interpolated = []

        # Interpolate matched boxes
        for last_idx, next_idx in matched_pairs:
            interp_box = self.interpolate_boxes(
                last_detections[last_idx], current_detections[next_idx], alpha
            )
            interpolated.append(interp_box)

        # Carry forward unmatched boxes from last frame
        for last_idx in unmatched_last:
            interpolated.append(last_detections[last_idx])

        return interpolated

    def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        store_id: str = "unknown",
    ):
        """Processes a single frame from a camera.

        Args:
            camera_id (str): The ID of the camera.
            frame (np.ndarray): The frame to process.
            frame_number (int): The frame number.
            timestamp (float): The timestamp of the frame.
            store_id (str): The store ID associated with the camera.
        """

        camera_state = self.get_camera_state(camera_id)

        self.latency_stats.total += 1

        is_yolo_frame = frame_number % config.YOLO_FRAMES_TO_INTERPOLATE == 0

        if is_yolo_frame:

            start_time = time.time()
            detections = self.run_yolo_inference(frame)
            latency = time.time() - start_time

            camera_state.update_state(detections)

            self.latency_stats.add_latency(latency)

        else:
            camera_state.frames_since_last_yolo += 1
            if (
                len(camera_state.last_detections) > 0
                and len(camera_state.current_detections) > 0
            ):
                detections = self.interpolate_detections(
                    camera_state.last_detections,
                    camera_state.current_detections,
                    frames_in_between=camera_state.frames_since_last_yolo,
                )
            else:
                detections = []

        # Send results to theft detection service
        self.model_manager.send_to_theft_detection(
            camera_id, frame_number, frame, timestamp, detections, store_id
        )

        # Send to heatmap service (at 1 FPS on YOLO frames)
        if config.ENABLE_HEATMAP and is_yolo_frame:
            self.model_manager.send_to_heatmap(camera_id, frame, detections)
        
        # Send to people counting service (at full FPS)
        if config.ENABLE_PEOPLE_COUNTING:
            self.model_manager.send_to_people_counting(
                camera_id, frame_number, frame, timestamp, detections, store_id
            )

        # Log stats periodically
        if (
            self.latency_stats.inference_count > 0
            and self.latency_stats.inference_count % config.YOLO_STATS_INTERVAL == 0
        ):
            logger.info(f"YOLO Stats: {self.latency_stats}")
            self.latency_stats.reset_latency_stats()

    def cleanup(self):
        """Cleans up resources."""
        self.model_manager.cleanup()
        logger.info(f"YOLO Worker {self.worker_id}: Cleaned up resources.")

    def run(self):
        """Starts the YOLO processing loop."""
        logger.info("YOLO Processor: Starting processing loop...")

        self.load_yolo_model()

        try:
            while True:
                # Check if we should stop
                if self.stop_event and self.stop_event.is_set():
                    logger.info("YOLO Processor: Stop event received, shutting down...")
                    break

                try:
                    frame_data = self.frame_queue.get(timeout=1)
                except Exception:
                    continue

                self.process_frame(**frame_data)

        except KeyboardInterrupt:
            logger.info("YOLO Processor: Received KeyboardInterrupt, shutting down...")
        except Exception as e:
            logger.error(f"YOLO Processor: Encountered error: {e}")
        finally:
            self.cleanup()
            logger.info("YOLO Processor: Shutdown complete.")


def main():
    from multiprocessing import Queue

    frame_queue = Queue(maxsize=3000)

    yolo_processor = YoloInferenceService(frame_queue)
    yolo_processor.run()


if __name__ == "__main__":
    main()
