import os
import numpy as np
from queue import Queue
import logging
from typing import Dict, Any
import zlib
import base64
import json

from app import config

logger = logging.getLogger("post-process")

class Post_process:
    def __init__(self, q: Queue) -> None:
        """
        Initializes the Post_process object.
        
        Parameters:
            q (Queue): Queue for inter-thread communication.
        
        Processing:
            - Sets up queue and camera-level floormap and aisle info dictionaries.
        
        Returns:
            None
        """
        self.q = q
        # Store camera-level data
        self.camera_floormaps: Dict[str, np.ndarray] = {}  # {camera_id: floormap}
        self.camera_aisle_info: Dict[str, Dict[str, Any]] = {}  # {camera_id: aisle_info}
    
    def arr_to_str(self, heat_map: np.ndarray) -> str:
        """
        Converts numpy heatmap array to compressed base64 string.
        
        Parameters:
            heat_map (np.ndarray): Heatmap array
        
        Returns:
            str: Compressed and base64 encoded string
        """
        array_list = heat_map.tolist() if hasattr(heat_map, 'tolist') else heat_map
        json_str = json.dumps(array_list)
        json_bytes = json_str.encode('utf-8')
        compressed_bytes = zlib.compress(json_bytes)
        compressed_string = base64.b64encode(compressed_bytes).decode('utf-8')
        return compressed_string

    def image_postprocess(self, camera_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Processes per-camera heatmaps and aisle info.
        
        Parameters:
            camera_data (Dict): Dictionary of camera data
                {
                    "cam_001": {
                        "heatmap": np.ndarray,
                        "aisle_info": {},
                        "store_id": "store_123"
                    }
                }
        
        Processing:
            - Stores heatmap and aisle info per camera
        
        Returns:
            None
        """
        for camera_id, data in camera_data.items():
            heat = data.get("heatmap")
            aisle_info = data.get("aisle_info", {})
            
            if heat is None:
                heat = np.zeros((480, 640), dtype=np.float32)
            
            # Store heatmap directly by camera_id
            self.camera_floormaps[camera_id] = heat
            
            # Store aisle info directly by camera_id
            self.camera_aisle_info[camera_id] = aisle_info
        
        logger.info(f"Post-process: Processed {len(self.camera_floormaps)} cameras")

    def push_to_queue(self) -> None:
        """
        Pushes processed camera-level floormaps and aisle info to the shared queue.
        
        Processing:
            - Compresses heatmap to string before pushing
            - Creates payload: {camera_id: {"heatmap": compressed_string, "aisle_info": {}}}
        
        Returns:
            None
        """
        payload = {}
        for camera_id, floormap in self.camera_floormaps.items():
            aisle_data = self.camera_aisle_info.get(camera_id, {"aisles": {}})
            
            # Compress heatmap to string (like heat-map does)
            floormap_str = self.arr_to_str(floormap)
            
            payload[camera_id] = {
                "heatmap": floormap_str,  # Already compressed string
                "aisle_info": aisle_data
            }
        
        self.q.put_nowait(payload)
        logger.info(f"Post-process: Pushed {len(payload)} cameras to shared queue")
        
        # Clear dictionaries for next iteration
        self.camera_floormaps.clear()
        self.camera_aisle_info.clear()

