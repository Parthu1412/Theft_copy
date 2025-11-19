import os
import cv2
import logging

from app import config
from antmedia_service.webrtc_subscriber import AntMediaCamera
from app.config import CameraConfig
from typing import Union
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("CameraInit")


class CameraInit:
    """Initialize camera based on client type."""

    def __init__(self, config: CameraConfig) -> None:

        self.config = config

    def camera_init(self) -> Union[AntMediaCamera, cv2.VideoCapture]:
        """
        Initialize camera based on client type.

        Returns:
            Camera object of appropriate type
        """

        logger.info(f"Initializing Camera {self.config}")

        if self.config.client_type == "webrtc":

            video = AntMediaCamera(
                websocket_url=self.config.websocket_url,
                stream_id=self.config.url,
                buffer_size=config.BUFFER_SIZE,
            )

        else:
            # Default case - assume RTSP or other OpenCV-compatible source
            video = cv2.VideoCapture(self.config.url)

            if not video.isOpened():
                logger.error(f"Failed to open camera stream: {self.config.url}")
                raise ConnectionError("Camera Initialization Failed")

        return video
