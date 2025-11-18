import os
import logging
from typing import Dict

from app.config import CameraConfig
from app import config

logger = logging.getLogger("CameraConfigLoader")


class CameraConfigLoader:
    """Camera Configuration Loader."""

    def __init__(self):
        """Initialize the camera config loader."""
        self.camera_configs: Dict[int, CameraConfig] = {}

    def __str__(self):
        """String representation."""
        return f"Camera Configs: {self.camera_configs}"

    def load_camera_configs(self) -> Dict[int, CameraConfig]:
        """Load camera configurations from environment variables and return them."""

        n = config.TOTAL_CAMERAS

        logger.info(f"Camera Orchestrator: Loading {n} camera configurations")

        for i in range(1, n + 1):
            try:
                self.camera_configs[i] = CameraConfig(
                    id=os.getenv(f"CAMERA_ID_{i}", f"cam_{i}"),
                    index=i,
                    url=os.getenv(f"CAMERA_URL_{i}", ""),
                    client_type=os.getenv(f"CLIENT_TYPE_{i}", "rtsp"),
                    store_id=os.getenv(f"STORE_ID_{i}", config.STORE_ID),
                    websocket_url=os.getenv(f"WEBSOCKET_URL_{i}", ""),
                    moksa_camera_id=int(os.getenv(f"MOKSA_CAMERA_ID_{i}", "0")),
                )
            except ValueError as e:
                logger.error(f"Camera Orchestrator: Failed to load camera {i}: {e}")
                continue

        logger.info(
            f"Camera Orchestrator: Loaded {len(self.camera_configs)} camera configurations"
        )

        return self.camera_configs

