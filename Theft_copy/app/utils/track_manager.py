import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)

class TrackerManager:
    """A simple object tracker that assigns and maintains IDs for detections."""
    
    def __init__(self, camera_id: int, id_offset: int = 1000):
        """
        Initialize a simple tracker for a specific camera.
        
        Args:
            camera_id: Camera identifier
            id_offset: ID offset for this camera
        """
        self.camera_id = camera_id
        self.id_offset = id_offset
        self.next_id = camera_id * id_offset
        # Store previous detections as {id: (bbox, class_id, last_seen_time)}
        self.prev_detections: Dict[int, Tuple[List[int], int, float]] = {}
    
    def calc_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        # Get intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        return intersection_area / union_area
    
    def calc_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate center point distance between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Euclidean distance between centers
        """
        # Calculate centers
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Calculate Euclidean distance
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update_id(self, bbox: List[int], class_id: int) -> int:
        """
        Update tracking for a single detection and return its ID.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class identifier
            
        Returns:
            Tracking ID for this detection
        """
        current_time = time.time()
        
        # If no previous detections, assign new ID
        if not self.prev_detections:
            new_id = self.next_id
            self.next_id += 1
            self.prev_detections[new_id] = (bbox, class_id, current_time)
            return new_id
        
        # Find the best match among previous detections
        best_id = None
        best_score = 0.0
        min_distance = float('inf')
        
        for track_id, (prev_bbox, prev_class, last_seen) in self.prev_detections.items():
            # Skip if class doesn't match
            if prev_class != class_id:
                continue
            
            # Calculate IoU
            iou = self.calc_iou(bbox, prev_bbox)
            
            # Calculate center distance
            distance = self.calc_distance(bbox, prev_bbox)
            
            # Consider detection a match if IoU > 0.3 or distance < 50 and it's a better match than previous ones
            if ((iou > 0.1 or distance < 50) and 
                (iou > best_score or (iou == best_score and distance < min_distance))):
                best_id = track_id
                best_score = iou
                min_distance = distance
        
        # If a good match is found, update the detection
        if best_id is not None:
            self.prev_detections[best_id] = (bbox, class_id, current_time)
            return best_id
        
        # Otherwise, assign a new ID
        new_id = self.next_id
        self.next_id += 1
        self.prev_detections[new_id] = (bbox, class_id, current_time)
        return new_id
    
    def cleanup(self, max_age: float = 1.0) -> None:
        """
        Remove detections that haven't been seen for a while.
        
        Args:
            max_age: Maximum time (in seconds) to keep inactive detections
        """
        current_time = time.time()
        ids_to_remove = []
        
        for track_id, (_, _, last_seen) in self.prev_detections.items():
            if current_time - last_seen > max_age:
                ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            del self.prev_detections[track_id]
    
    def reset(self) -> None:
        """Reset the tracker, keeping only camera ID and offset."""
        self.prev_detections = {}
        # Don't reset next_id to maintain uniqueness across resets


class MultiCameraTracker:
    """Manages multiple simple trackers, one per camera."""
    
    def __init__(self, id_offset: int = 1000):
        """
        Initialize the multi-camera simple tracker.
        
        Args:
            id_offset: Base ID offset for each camera
        """
        self.trackers: Dict[int, TrackerManager] = {}
        self.id_offset = id_offset
    
    def get_tracker(self, camera_id: int) -> TrackerManager:
        """
        Get or create a tracker for a specific camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Tracker for the specified camera
        """
        if camera_id not in self.trackers:
            self.trackers[camera_id] = TrackerManager(camera_id, self.id_offset)
        
        return self.trackers[camera_id]
    
    def update_id(self, camera_id: int, bbox: List[int], class_id: int) -> int:
        """
        Get tracking ID for a detection in a specific camera.
        
        Args:
            camera_id: Camera identifier
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class identifier
            
        Returns:
            Tracking ID
        """
        tracker = self.get_tracker(camera_id)
        return tracker.update_id(bbox, class_id)
    
    def cleanup_all(self, max_age: float = 1.0) -> None:
        """
        Clean up old detections for all cameras.
        
        Args:
            max_age: Maximum time to keep inactive detections
        """
        for tracker in self.trackers.values():
            tracker.cleanup(max_age)
    
    def reset_camera(self, camera_id: int) -> None:
        """
        Reset a specific camera's tracker.
        
        Args:
            camera_id: Camera identifier
        """
        if camera_id in self.trackers:
            self.trackers[camera_id].reset()
    
    def reset_all(self) -> None:
        """Reset all camera trackers."""
        for tracker in self.trackers.values():
            tracker.reset()