import os
import socket
from dataclasses import dataclass


# Camera
STORE_ID = os.getenv("STORE_ID", "123")
CAMERA_IP = os.getenv("CAMERA_IP", "192.168.1.1")
CAMERA_NO = os.getenv("CAMERA_NO", "102")
RTSP_URL = os.getenv("CAMERA_URL", None)
CLIENT_TYPE = os.getenv("CLIENT_TYPE", "rtsp")
RABBITMQ_CAMERAID = os.getenv("RABBITMQ_CAMERAID", None)
FRAME_LENGTH = int(os.getenv("FRAME_LENGTH", 100))
GPU_LIMIT = int(os.getenv("GPU_LIMIT", 8000))

MODEL_PATH = os.getenv("MODEL_PATH", None)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)
AWS_BUCKET = os.getenv("AWS_BUCKET", None)
AWS_OBJECT_NAME = os.getenv("AWS_OBJECT_NAME", "")
AWS_REGION = os.getenv("KAFKA_AWS_REGION", "us-east-2")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", None)
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", None)
RABBITMQ_USER = os.getenv("RABBITMQ_USER", None)
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", None)
RABBITMQ_QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", None)
RABBITMQ_CAMERAID = os.getenv("RABBITMQ_CAMERAID", None)

RABBITMQ_HOST_PRODUCER = os.getenv("RABBITMQ_HOST_PRODUCER", "")
RABBITMQ_PORT_PRODUCER = int(os.getenv("RABBITMQ_PORT_PRODUCER", 0))
RABBITMQ_USER_PRODUCER = os.getenv("RABBITMQ_USER_PRODUCER", "")
RABBITMQ_PASS_PRODUCER = os.getenv("RABBITMQ_PASS_PRODUCER", "")
RABBITMQ_QUEUE_NAME_PRODUCER = "theft_" + os.getenv("STORE_ID", "000")
DEFAULT_MESSAGE_RMQ = os.getenv("DEFAULT_MESSAGE_RMQ", "True")
STORE_MESSAGE_RMQ = os.getenv("STORE_MESSAGE_RMQ", "True")

# YOLO Configuration
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))
YOLO_FRAMES_TO_INTERPOLATE = int(os.getenv("YOLO_FRAMES_TO_INTERPOLATE", 5))
YOLO_STATS_INTERVAL = int(os.getenv("YOLO_STATS_INTERVAL", 300))
NUM_YOLO_WORKERS = int(os.getenv("NUM_YOLO_WORKERS", 1))  # Number of parallel YOLO processes

# Theft Inference Configuration
ENABLE_THEFT_FILTERING = os.getenv("ENABLE_THEFT_FILTERING", "false").lower() == "true"
THEFT_WINDOW_SIZE = int(os.getenv("THEFT_WINDOW_SIZE", 25))
THEFT_CONT_PREDS = int(os.getenv("THEFT_CONT_PREDS", 3))
THEFT_THRESHOLD = float(os.getenv("THEFT_THRESHOLD", 0.7))
THEFT_INFERENCE_BATCH = int(os.getenv("THEFT_INFERENCE_BATCH", 100))
THEFT_FRAMES_BUFFER = int(os.getenv("THEFT_FRAMES_BUFFER", 300))

# API Configuration
THEFT_API_ENDPOINT = os.getenv("THEFT_API_ENDPOINT", "http://localhost:8004/submitTheft")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
STREAM_BASE_URL = os.getenv("STREAM_BASE_URL", "http://100.85.96.15:5080/WebRTCAppEE/streams")
API_REQUEST_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", "1.0"))

# ---------------------------
# Heatmap Configuration
# ---------------------------
ENABLE_HEATMAP = os.getenv("ENABLE_HEATMAP", "false").lower() == "true"
HEAT_INTERVAL = int(os.getenv("HEAT_INTERVAL", 300))  # seconds
HEAT_MAP_INTENSITY = float(os.getenv("HEAT_MAP_INTENSITY", 0.0001))

# ---------------------------
# People Counting Configuration
# ---------------------------

ENABLE_PEOPLE_COUNTING = os.getenv("ENABLE_PEOPLE_COUNTING", "false").lower() == "true"
PEOPLE_COUNTING_PORT = 5596  # Base port for people counting (5596, 5597, 5598...)
KAFKA_PEOPLE_COUNTING_TOPIC = os.getenv("KAFKA_PEOPLE_COUNTING_TOPIC", "people-counting")

# ---------------------------
# Store + Camera Configuration
# ---------------------------
TOTAL_CAMERAS = int(os.getenv("TOTAL_CAMERAS", 0))
TOTAL_STORES = int(os.getenv("TOTAL_STORES", 0))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 50))

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "theft_topic")
KAFKA_HEAT_MAP_TOPIC = os.getenv("KAFKA_HEAT_MAP_TOPIC", "heatmap")
KAFKA_AISLE_INFO_TOPIC = os.getenv("KAFKA_AISLE_INFO", "aisle-count")
KAFKA_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", socket.gethostname())
AISLE_INFO_ENABLED = os.getenv("AISLE_INFO", "True").lower() == "true"


@dataclass
class CameraConfig:
    """Camera configuration data."""

    id: str
    index: int
    url: str
    client_type: str
    store_id: str
    websocket_url: str
    moksa_camera_id: int = 0

    def __post_init__(self):
        if not self.id:
            raise ValueError("Camera ID cannot be empty")

        if not self.url:
            raise ValueError(f"Camera URL cannot be empty for camera {self.id}")

        if self.client_type not in ["webrtc", "rtsp"]:
            raise ValueError(
                f"Invalid client type {self.client_type} for camera {self.id}"
            )

        if not self.store_id:
            raise ValueError(f"Store ID cannot be empty for camera {self.id}")

        if self.client_type == "webrtc" and not self.websocket_url:
            raise ValueError(
                f"WebSocket URL required for WebRTC camera {self.id} but not provided"
            )

    def __str__(self) -> str:
        return (
            f"CameraConfig(id={self.id}, index={self.index}, url={self.url}, "
            f"client_type={self.client_type}, store_id={self.store_id}, "
            f"websocket_url={self.websocket_url}, moksa_camera_id={self.moksa_camera_id})"
        )