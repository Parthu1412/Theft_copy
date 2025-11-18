import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from app import tzinfo
from app import config
from app.utils.message import PeopleCountingMessage
from app.utils.track_manager import MultiCameraTracker


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("YOLOv8_TRT")

class PeopleCountingDetection:
    def __init__(
        self,
        kafka_producer=None,
        region_points: list = None,
        camera_id: str = "0",
        camera_index: int = 0,
        moksa_camera_id: int = 0,
        store_id: str = "123",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> None:
        self.counting_region = Polygon(region_points) if region_points else None
        self.kafka_producer = kafka_producer
        self.track_history = defaultdict(list)
        self.counting_history = defaultdict(set)
        self.reverse_direction = getattr(config, 'reverse_direction', False)
        self.outside_region_timer = defaultdict(int)
        self.max_outside_time = 30
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.moksa_camera_id = moksa_camera_id
        self.store_id = store_id
        self.conf_thresh = conf_thresh

        # Tracking
        self.tracker = MultiCameraTracker()
        # No TensorRT engine needed - using detections from YOLO
        logger.info(f"Redis_Detection initialized for camera {camera_id}")


    async def detect(self, detections, verbose: bool = False) -> None:
        """
        Process detections from Redis instead of running inference
        
        Args:
            detections: List of detection dictionaries from Redis
                    Format: [{'box': [x1, y1, x2, y2]}, ...]
        """
        logger.debug("Processing Redis detections")
        
        # Collect bounding boxes
        track_ids = []
        for detection in detections:
            if "box" in detection:  # only use bbox
                bbox = detection['box']
                bbox_int = [int(x) for x in bbox]  # ensure ints for tracker
                track_id = self.tracker.update_id(self.camera_index, bbox_int, 0)  # keep score=0
                track_ids.append(track_id)

                # save trajectory history
                self.track_history[track_id].append(
                    ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                )
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)


        # Counting logic (same as before)
        for track_id, track_line in self.track_history.items():
            if len(track_line) < 2:
                continue
            if self.counting_region.contains(Point(track_line[-1])):
                direction = track_line[-1][1] - np.mean([point[1] for point in track_line[:-1]])
                if direction > 0:
                    detected_direction = "OUT" if self.reverse_direction else "IN"
                else:
                    detected_direction = "IN" if self.reverse_direction else "OUT"
                if detected_direction not in self.counting_history[track_id]:
                    self.counting_history[track_id].add(detected_direction)
                    logger.info(f"Detected direction: {detected_direction}")
                    await self.produce_message(detected_direction)
            else:
                self.outside_region_timer[track_id] += 1
                if self.outside_region_timer[track_id] >= self.max_outside_time:
                    self.counting_history.pop(track_id, None)
                    self.outside_region_timer.pop(track_id, None)


    async def produce_message(self, _type: str) -> None:
        message = PeopleCountingMessage(
            camera_id=self.camera_id,
            going_in=1 if _type == "IN" else 0,
            going_out=1 if _type == "OUT" else 0,
            timestamp=datetime.now(tz=tzinfo) if tzinfo else datetime.now(),
            model_version="v1.0.0",
            moksa_camera_id=self.moksa_camera_id,
            store_id=int(self.store_id) if isinstance(self.store_id, str) and self.store_id.isdigit() else 0,
        )
        logger.debug(f"Attempting to produce message: {message}")
        if self.kafka_producer:
            try:
                await self.kafka_producer.produce(
                    topic=config.KAFKA_PEOPLE_COUNTING_TOPIC, message=message
                )
                logger.info(f"Message produced successfully: {message}")
            except Exception as e:
                logger.error(f"Failed to produce message: {e}")
        else:
            logger.warning("Kafka producer not initialized. Message not sent.")