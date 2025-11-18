from shapely.geometry import Polygon
import logging
from typing import Dict, List, Tuple, Any, Optional
from app import config

logger = logging.getLogger("POLYGON RETRIEVAL") 

class ProcessingPolygons:
    def __init__(self) -> None:
        """
        Initializes the ProcessingPolygons object.

        Parameters:
            None

        Processing:
            - Sets up the dictionary to store polygons for all cameras.

        Returns:
            None
        """
        self.all_camera_polygons: Dict[int,Dict[str, Dict[str, Any]]] = {}

    def load_from_config(self) -> None:
        """
        Loads polygons from the configuration file for all cameras.

        Parameters:
            None

        Processing:
            - Iterates through camera indices.
            - Retrieves polygon and aisle info strings from config.
            - Calls parse_camera_polygons for each camera.

        Returns:
            None
        """
        camera_count = int(config.TOTAL_CAMERAS)
            
        for i in range(1, camera_count + 1):
            cam_index = str(i)
            real_camera_id = getattr(config, f"RABBITMQ_CAMERA_ID_{cam_index}", None)
            polygon_str = getattr(config, f"POLYGON_CAM_{cam_index}", None)
            polygon_info_str = getattr(config, f"POLYGON_INFO_CAM_{cam_index}", None)

            if not (real_camera_id and polygon_str and polygon_info_str):
                logger.warning(f"Missing config for camera index {cam_index}")
                continue

        self.parse_camera_polygons(polygon_str, polygon_info_str)

    def parse_camera_polygons(
            self,
            camera_id: str,
            polygon_str: str,
            polygon_info_str: str,
    ) -> None:
        """
        Parses polygon and aisle info strings for a camera.

        Parameters:
            camera_id (str): Camera identifier.
            polygon_str (str): Polygon coordinates string.
            polygon_info_str (str): Aisle info string.

        Processing:
            - Splits polygon and info strings.
            - Creates Polygon objects and stores them with aisle names.

        Returns:
            None
        """
            
        polygon_sections = polygon_str.strip().split("#")
        info_sections = polygon_info_str.strip().split("#")


        if len(polygon_sections) != len(info_sections):
            logger.error(f"Polygon and info sections mismatch for camera {camera_id}.")
            return


            
        aisle_data : Dict[str, Dict[str, Any]] = {}


        for coords, aisle_info in zip(polygon_sections, info_sections):
            try: 
                _, aisle_name = aisle_info.split(":")
                coord_pairs = coords.split(",")
                points: List[Tuple[int, int]] = [tuple(map(int, pt.split("-")))for pt in coord_pairs]
                aisle_data[aisle_name] = {
                    "polygon": Polygon(points),
                    "coordinates": points,
                }
                
            except Exception as e:
                logger.error(f"Error parsing polygon data for camera {camera_id}: {aisle_info}, {coords} — {e}")
                continue
            
        self.all_camera_polygons[camera_id] = aisle_data    


        
    def get_camera_polygons(self, camera_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve polygons for a specific camera ID.
            
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary of aisle polygons and their coordinates, or None if not found.
        """
        return self.all_camera_polygons.get(camera_id, {} )
        
    def get_polygons(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all camera polygons.
            
        Returns:
            Dictionary of all camera polygons.
        """
        return self.all_camera_polygons

