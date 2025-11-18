import time
import numpy as np
import logging
from shapely.geometry import Polygon, Point
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import cv2
 
from app import config
 
logger = logging.getLogger("AisleTracker")
 
class AisleTracker:
    def __init__(self, polygon_dict: Dict[str, Dict[str, Any]], store_id: str, camera_id: str):
        """
        Initializes the AisleTracker object.

        Parameters:
            polygon_dict (Dict[str, Dict[str, Any]]): Dictionary of aisle polygons.
            store_id (str): Store identifier.
            camera_id (str): Camera identifier.

        Processing:
            - Sets up tracking, polygon data, and statistics containers.

        Returns:
            None
        """
        self.camera_polygons = polygon_dict
        # Ensure both store_id and camera_id are strings, handling both string and int inputs
        self.store_id = str(store_id) if store_id is not None else ""
        self.camera_id = str(camera_id) if camera_id is not None else ""
        
        # Log the types for debugging
        logger.debug(f"AisleTracker initialized with store_id: {self.store_id} (type: {type(self.store_id)}), camera_id: {self.camera_id} (type: {type(self.camera_id)})")
 
        print(f"Loaded {len(self.camera_polygons)} polygons for camera {self.camera_id} in store {self.store_id}")
 
        self.last_reset_time: float = time.time()
        self.reid_reset_interval: int = int(config.HEAT_INTERVAL) * 2
 
        self.polygon_stats = defaultdict(self._init_polygon_stats)
        self.aisle_final_counts = defaultdict(int)
        self.cumulative_time_spent = defaultdict(float)
        self.cumulative_final_counts = defaultdict(int)
 
        self.heatmap: Optional[np.ndarray] = None
        self.img_width = 640
        self.img_height = 480
 
    def _init_polygon_stats(self):
        """
        Initializes statistics for a polygon.

        Parameters:
            None

        Processing:
            - Creates a dictionary for tracking unique persons, time spent, and counts.

        Returns:
            dict: Initialized statistics dictionary.
        """
        return {
            'unique_persons': set(),
            'total_time_spent': 0.0,
            'start_time': None,
            'current_count': 0
        }
 
    def handle_camera_switch(self, frame_shape: Tuple[int, int]) -> float:
        """
        Handles camera switch and heatmap initialization.

        Parameters:
            frame_shape (Tuple[int, int]): Shape of the input frame.

        Processing:
            - Updates image dimensions.
            - Initializes heatmap if needed.
            - Resets tracker if interval elapsed.

        Returns:
            float: Current timestamp.
        """
        current_time = time.time()
        self.img_height, self.img_width = frame_shape[:2]
 
        if self.heatmap is None:
            self.heatmap = np.zeros((self.img_height, self.img_width), dtype=np.float32)
 
        return current_time
 
    def extract_person_boxes(self, results: Any) -> List[Tuple[List[int], float]]:
        """
        Extracts person bounding boxes from model results.

        Parameters:
            results (Any): Model output containing detection boxes.

        Processing:
            - Parses detection results for person class and confidence.

        Returns:
            List[Tuple[List[int], float]]: List of bounding boxes and confidence scores.
        """
        if not results or not hasattr(results[0], 'boxes'):
            return []
 
        boxes = []
        for box in results[0].boxes:
            if not hasattr(box, 'xyxy') or box.xyxy is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 1 and conf >= float(getattr(config, 'PERSON_CONFIDENCE', 0.5)):
                boxes.append(([x1, y1, x2, y2], conf))
        return boxes
 
    def process_person(self, person_bbox: List[int], current_time: float, aisle_counts: Dict[str, int]):
        """
        Processes a detected person for aisle tracking.

        Parameters:
            person_bbox (List[int]): Bounding box of the person.
            current_time (float): Current timestamp.
            aisle_counts (Dict[str, int]): Dictionary of aisle counts.

        Processing:
            - Updates tracker and statistics for each aisle.
            - Updates heatmap if person is inside polygon.

        Returns:
            None
        """
        person_poly = self._get_polygon(person_bbox)
        center_point = self._get_bbox_bottom_center_point(person_bbox)
 
        for aisle_name, poly_data in self.camera_polygons.items():
            poly = poly_data['polygon']
            if person_poly.area == 0:
                continue

 
            stats = self.polygon_stats[aisle_name]
 
           
            if poly.contains(center_point):
                aisle_counts[aisle_name] += 1
 
                if stats['start_time'] is None:
                    stats['start_time'] = current_time
 
                if self.heatmap is not None:
                    self._update_heatmap(center_point, poly)

            
            
 
    def update_time_and_counts(self, aisle_counts: Dict[str, int], current_time: float):
        """
        Updates time spent and counts for each aisle.

        Parameters:
            aisle_counts (Dict[str, int]): Dictionary of aisle counts.
            current_time (float): Current timestamp.

        Processing:
            - Updates cumulative time and counts.
            - Tracks changes in person counts.

        Returns:
            None
        """
        for aisle_name, stats in self.polygon_stats.items():
            current_count = aisle_counts.get(aisle_name, 0)
            previous_count = stats.get('current_count', 0)

            logger.info(f"aisle_name: {aisle_name}, current_count: {current_count}, previous_count: {previous_count}")
            
            stats['total_time_spent'] += current_count
            self.cumulative_time_spent[aisle_name] += current_count
            stats['start_time'] = current_time

            if previous_count > 0 and current_count < previous_count:
                delta = previous_count - current_count
                self.aisle_final_counts[aisle_name] += delta
                self.cumulative_final_counts[aisle_name] += delta
               
            stats['current_count'] = current_count
 
    def process_frame(self, frame: np.ndarray, detections:List) -> Dict[str, Any]:
        """
        Processes a frame for aisle statistics.

        Parameters:
            frame (np.ndarray): Input image frame.
            detections (List): List of detection bounding boxes.

        Returns:
            Dict[str, Any]: Aisle statistics for the frame.
        """
        try:

            logger.info(f"- Received frame from camera_id: {self.camera_id} and Detections: {len(detections)}")
           
            # 🔹 Normal frame processing
            current_time = self.handle_camera_switch(frame.shape)      
            aisle_counts = defaultdict(int)
            for bbox in detections:
                self.process_person(bbox, current_time, aisle_counts)

            self.update_time_and_counts(aisle_counts, current_time)
            return self.get_statistics()

        except Exception as e:
            logger.error(f"AisleTracker Error (Store: {self.store_id}, Camera: {self.camera_id}): {e}")
            return {str(self.store_id): {"entries": []}}  # 🛡️ Enforced str key

 
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gets cumulative statistics for all aisles.

        Parameters:
            None

        Processing:
            - Aggregates counts and time spent for each aisle.

        Returns:
            Dict[str, Any]: Statistics dictionary for all aisles.
        """
        aisle_data = {}
 
        for aisle_name in self.camera_polygons.keys():
            aisle_data[aisle_name] = {
                "count": self.cumulative_final_counts.get(aisle_name, 0),
                "time": round(self.cumulative_time_spent.get(aisle_name, 0.0), 2)
            }
 
        return {
            str(self.camera_id): 
            {
                "aisles": aisle_data
            }
        }
 
    def _get_polygon(self, bbox: List[int]) -> Polygon:
        """
        Converts a bounding box to a shapely Polygon.

        Parameters:
            bbox (List[int]): Bounding box coordinates.

        Processing:
            - Creates a polygon from bbox corners.

        Returns:
            Polygon: Shapely polygon object.
        """
        x1, y1, x2, y2 = bbox
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
 
    def _get_bbox_bottom_center_point(self, bbox: List[int]) -> Point:
        """
        Gets the bottom center point of a bounding box.

        Parameters:
            bbox (List[int]): Bounding box coordinates.

        Processing:
            - Calculates bottom center coordinates.

        Returns:
            Point: Shapely point object.
        """
        x1, y1, x2, y2 = bbox
        return Point((x1 + x2) / 2, y2)
 
    def _update_heatmap(self, center: Point, polygon: Polygon):
        """
        Updates the heatmap for a detected person.

        Parameters:
            center (Point): Center point of the person.
            polygon (Polygon): Aisle polygon.

        Processing:
            - Creates masks and updates heatmap intensity.

        Returns:
            None
        """
        cx, cy = int(center.x), int(center.y)
        size = 20
        x1 = max(cx - size // 2, 0)
        y1 = max(cy - size // 2, 0)
        x2 = min(cx + size // 2, self.img_width)
        y2 = min(cy + size // 2, self.img_height)
 
        poly_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        rect_mask = np.zeros_like(poly_mask)
 
        pts = np.array([list(polygon.exterior.coords)], dtype=np.int32)
        cv2.fillPoly(poly_mask, pts, 1)
        cv2.rectangle(rect_mask, (x1, y1), (x2, y2), 1, thickness=-1)
 
        intersection = cv2.bitwise_and(poly_mask, rect_mask)
        self.heatmap += intersection.astype(np.float32) * config.HEAT_MAP_INTENSITY
 
    def detect(self, image, detections):
        """
        Runs detection and returns heatmap and aisle info.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Heatmap and aisle info.
        """
      
        aisle_info = self.process_frame(image,detections)
        return self.heatmap, aisle_info
 
    def reset_heatmap(self):
        """
        Resets the heatmap to zero.

        Parameters:
            None

        Processing:
            - Fills heatmap array with zeros.

        Returns:
            None
        """
        if self.heatmap is not None:
            self.heatmap.fill(0)
 
    def reset_statistics(self):
        """
        Resets all aisle statistics and heatmap.

        Parameters:
            None

        Processing:
            - Clears all statistics and resets tracker and heatmap.

        Returns:
            None
        """
        # self.polygon_stats.clear()
        self.aisle_final_counts.clear()
        self.cumulative_final_counts.clear()
        self.cumulative_time_spent.clear()
        if self.heatmap is not None:
            self.heatmap.fill(0)

        logger.info("Reset all aisle stats and heatmap.")
