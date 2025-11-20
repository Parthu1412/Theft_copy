#!/usr/bin/env python3
"""
Camera Data Collection Orchestrator.
 
Reads per-camera configuration from environment variables and performs:
1. People Count snapshot upload
2. Heat Map snapshot upload
3. Re-ID hourly video chunk recording & upload
"""
 
import datetime as dt
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
 
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
 
from app.config import CameraConfig
from app.utils.camera_intialize import CameraInit
from app.utils.aws import S3Client
 
#Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
 
 
# Custom Exceptions
class DataCollectionError(Exception):
    """Base exception for data collection errors."""
 
 
class APIError(DataCollectionError):
    """Raised when API interactions fail."""
 
 
class CameraError(DataCollectionError):
    """Raised when camera operations fail."""
 
 
# Configuration & Data Structures
@dataclass(frozen=True)
class AppSettings:
    """Centralized application configuration."""
 
    api_login_url: str = os.getenv("API_LOGIN_URL", "http://alpha.api.moksa.ai/auth/login")
    api_store_url: str = os.getenv("API_STORE_URL", "http://alpha.api.moksa.ai/store/getAllStoresForDropdown")
    api_email: Optional[str] = os.getenv("API_EMAIL")
    api_password: Optional[str] = os.getenv("API_PASSWORD")
    save_in_s3: bool = field(default_factory=lambda: os.getenv("SAVE_IN_S3", "true").lower() in ("true", "1", "yes"))
    save_to_local: bool = field(default_factory=lambda: os.getenv("SAVE_TO_LOCAL", "true").lower() in ("true", "1", "yes"))
    re_id_interval: int = int(os.getenv("RE_ID_INTERVAL_SECONDS", "60"))
    aws_object_prefix: str = os.getenv("AWS_OBJECT_NAME", "")
    root_data_dir: Path = Path("Data_Collection")
 
 
@dataclass
class StoreInfo:
    """Store details."""
    id: str
    name: str
    clean_name: str
 
 
# Helper Services
 
class StoreAPIClient:
    """
    Handles authentication and retrieval of store details from the API.
    """
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.token: Optional[str] = None
 
    def _login(self) -> None:
        """Authenticates with the API to retrieve a bearer token."""
        if not self.settings.api_email or not self.settings.api_password:
            logger.warning("API credentials missing. Using Store IDs instead of names.")
            return
 
        try:
            payload = {"email": self.settings.api_email, "password": self.settings.api_password}
            resp = requests.post(self.settings.api_login_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
 
            # Handle various token locations in response
            self.token = data.get("token") or data.get("accessToken")
            if not self.token and "data" in data and isinstance(data["data"], dict):
                self.token = data["data"].get("token")
 
            if not self.token:
                raise APIError("Login successful but no token found in response.")
           
            logger.info("API Login successful.")
 
        except requests.RequestException as e:
            logger.exception(f"API Login failed: {e}")
            raise APIError("Failed to authenticate.") from e
 
    def fetch_store_map(self) -> Dict[str, str]:
        """
        Fetches all stores and maps ID to Sanitized Name.
        Return a Dict
        """
        if not self.token:
            try:
                self._login()
            except APIError:
                return {}
 
        if not self.token:
            return {}
 
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            resp = requests.get(self.settings.api_store_url, headers=headers, timeout=10)
            resp.raise_for_status()
           
            json_data = resp.json()
            stores_list = self._extract_list_from_response(json_data)
 
            store_map = {}
            for store in stores_list:
                if not isinstance(store, dict):
                    continue
                s_id = str(store.get("id"))
                s_name = store.get("name", f"Unknown_{s_id}")
                store_map[s_id] = self._sanitize_filename(s_name)
           
            logger.info(f"Cached names for {len(store_map)} stores.")
            return store_map
 
        except requests.RequestException as e:
            logger.exception(f"Failed to fetch store list: {e}")
            return {}
 
    def _extract_list_from_response(self, json_data: Any) -> List[Any]:
        """Normalizes inconsistent API nested list responses."""
        if isinstance(json_data, list):
            return json_data
        if isinstance(json_data, dict):
            if "data" in json_data:
                inner = json_data["data"]
                if isinstance(inner, list):
                    return inner
                if isinstance(inner, dict) and "data" in inner and isinstance(inner["data"], list):
                    return inner["data"]
            if "stores" in json_data:
                return json_data["stores"]
        logger.warning("Could not parse store list structure.")
        return []
 
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Replaces special chars with underscores."""
        return re.sub(r'[^a-zA-Z0-9-]', '_', str(name))
 
 
class CameraWorker(threading.Thread):
    """
    Handles video capture, snapshots, and chunked video recording for a specific camera.
    """
 
    def __init__(
        self,
        config: CameraConfig,
        settings: AppSettings,
        features: Dict[str, bool],
        s3_client: S3Client,
        store_folder_name: str,
    ) -> None:
        """
        Initialize the camera worker.
        """
        super().__init__(name=f"CamWorker-{config.id}", daemon=True)
        self.cfg = config
        self.settings = settings
        self.features = features
        self.s3 = s3_client
        self.store_folder = store_folder_name
       
        self.stop_event = threading.Event()
        self.cap: Optional[Any] = None  # cv2.VideoCapture
        self.video_writer: Optional[Any] = None # cv2.VideoWriter
        self.segment_start_time: Optional[float] = None
   
    def _init_camera(self) -> bool:
        """Initializes the camera connection."""
        try:
            cam_init = CameraInit(self.cfg)
            self.cap = cam_init.camera_init()
            if self.cap is None:
                raise CameraError(f"CameraInit returned None for {self.cfg.url}")
            logger.info(f"Camera {self.cfg.id} connected.")
            return True
        except Exception as e:
            logger.exception(f"Camera {self.cfg.id}: Initialization failed.")
            return False
 
    def _get_local_path(self, sub_folder: str, filename: str) -> Path:
        """Constructs the local file path."""
        return (
            self.settings.root_data_dir
            / self.store_folder
            / f"camera_{self.cfg.id}"
            / sub_folder
            / filename
        )
 
    def _manage_file_persistence(self, file_path: Path) -> None:
        """
        Uploads to S3 if enabled and manages local file retention.
        """
        uploaded = False
        if self.settings.save_in_s3:
            try:
                # Calculate object name relative to Root Directory
                relative_path = file_path.relative_to(self.settings.root_data_dir)
                object_name = f"{self.settings.aws_object_prefix}/{relative_path}".strip("/")
                object_name = object_name.replace("//", "/") # Sanitize
 
                url = self.s3.upload_file_and_get_direct_url(str(file_path), object_name)
                if url:
                    logger.info(f"Uploaded: {file_path.name}")
                    uploaded = True
                else:
                    logger.error(f"Upload returned None for {file_path.name}")
            except Exception as e:
                logger.exception(f"Failed to upload {file_path}: {e}")
 
        # Local Cleanup logic
        if not self.settings.save_to_local:
            try:
                file_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete local file {file_path}: {e}")
        elif not uploaded and self.settings.save_in_s3:
             logger.warning(f"Persisted locally (Upload failed): {file_path}")
 
    def _take_snapshot(self, frame: np.ndarray, tag: str) -> None:
        """Saves and uploads a single frame snapshot."""
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        folder_map = {"people": "people_count", "heatmap": "heat_map"}
        sub_folder = folder_map.get(tag, "misc")
       
        filename = f"{timestamp}_{self.cfg.id}_{tag}.jpg"
        file_path = self._get_local_path(sub_folder, filename)
       
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(file_path), frame)
            self._manage_file_persistence(file_path)
        except Exception as e:
            logger.exception(f"Snapshot failed for {tag}")
 
    def _start_video_segment(self) -> None:
        """Starts a new video recording segment."""
        self._close_video_writer()
       
        self.segment_start_time = time.time()
        ts_str = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"{ts_str}_inprogress_{self.cfg.id}_reid.mp4"
        file_path = self._get_local_path("re_id", filename)
 
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # cv2 specific setup
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(str(file_path), fourcc, 15.0, (640, 480))
            logger.debug(f"Started segment: {filename}")
        except Exception as e:
            logger.exception(f"Failed to start video segment: {e}")
            self.video_writer = None
 
    def _finalize_video_segment(self) -> None:
        """Stops recording, renames the file with end-timestamp, and uploads."""
        if not self.video_writer or not self.segment_start_time:
            return
 
        self._close_video_writer()
       
        end_time = time.time()
        start_ts = dt.datetime.utcfromtimestamp(self.segment_start_time).strftime("%Y%m%dT%H%M%S")
        end_ts = dt.datetime.utcfromtimestamp(end_time).strftime("%Y%m%dT%H%M%S")
       
        directory = self._get_local_path("re_id", "").parent
       
        # Find the temp file
        try:
            temp_files = list(directory.glob(f"{start_ts}_inprogress_{self.cfg.id}_reid.mp4"))
            if temp_files:
                temp_path = temp_files[0]
                final_name = f"{start_ts}_{end_ts}_{self.cfg.id}_reid.mp4"
                final_path = directory / final_name
               
                temp_path.rename(final_path)
                self._manage_file_persistence(final_path)
        except Exception as e:
            logger.exception("Failed to finalize video segment.")
 
    def _close_video_writer(self) -> None:
        """Safely releases the video writer."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
 
    def run(self) -> None:
        """Main thread loop."""
        if not self._init_camera():
            return
 
        # Warmup & Initial Snapshots
        logger.info(f"Camera {self.cfg.id}: Warming up...")
        first_frame = None
       
        # Attempt to get first frame
        for _ in range(200):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                first_frame = cv2.resize(frame, (640, 480))
                break
            time.sleep(0.1)
 
        if first_frame is not None:
            if self.features.get("PEOPLE_COUNT"):
                self._take_snapshot(first_frame, "people")
            if self.features.get("HEAT_MAP"):
                self._take_snapshot(first_frame, "heatmap")
        else:
            logger.error(f"Camera {self.cfg.id}: Failed to capture initial frame.")
 
        # Start Re-ID Recording
        if self.features.get("RE_ID"):
            self._start_video_segment()
 
        consecutive_failures = 0
       
        # Main Loop
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
           
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures % 100 == 0:
                    logger.warning(f"Camera {self.cfg.id}: High frame failure count ({consecutive_failures})")
                # Non-blocking wait
                if self.stop_event.wait(timeout=0.05):
                    break
                continue
 
            consecutive_failures = 0
           
            # Resize once for processing
            try:
                frame_resized = cv2.resize(frame, (640, 480))
            except cv2.error:
                continue
 
            # Handle Re-ID Video
            if self.features.get("RE_ID") and self.video_writer:
                self.video_writer.write(frame_resized)
               
                # Check interval
                if (time.time() - (self.segment_start_time or 0)) >= self.settings.re_id_interval:
                    self._finalize_video_segment()
                    self._start_video_segment()
 
            # CPU yield
            time.sleep(0.001)
 
        # Cleanup
        if self.features.get("RE_ID"):
            self._finalize_video_segment()
       
        if self.cap:
            self.cap.release()
       
        logger.info(f"Camera {self.cfg.id}: Worker stopped.")
 
    def stop(self) -> None:
        """Signals the thread to stop."""
        self.stop_event.set()
 
 
# Orchestrator
 
def load_cameras_from_env() -> Dict[int, Dict[str, Any]]:
    """
    Parses environment variables to build camera configurations.
    """
    cameras = {}
    total_cameras = int(os.getenv("TOTAL_CAMERAS", "0") or 0)
   
    # Auto-discovery if TOTAL_CAMERAS is not explicitly set
    if total_cameras <= 0:
        for i in range(1, 51):
            if os.getenv(f"CAMERA_ID_{i}"):
                total_cameras = max(total_cameras, i)
 
    for i in range(1, total_cameras + 1):
        cam_id = os.getenv(f"CAMERA_ID_{i}")
        url = os.getenv(f"CAMERA_URL_{i}")
       
        if not cam_id or not url:
            continue
 
        cameras[i] = {
            "id": cam_id,
            "index": i,
            "url": url,
            "client_type": os.getenv(f"CLIENT_TYPE_{i}", os.getenv("CLIENT_TYPE", "rtsp")),
            "store_id": os.getenv(f"STORE_ID_{i}", os.getenv("STORE_ID", "0")),
            "websocket_url": os.getenv(f"WEBSOCKET_URL_{i}", ""),
            "moksa_camera_id": int(os.getenv(f"CAMERA_ID_{i}", "0") or 0),
        }
    return cameras
 
def get_feature_flags(index: int) -> Dict[str, bool]:
    """Helper to check boolean env vars for specific camera index."""
    def check(key: str) -> bool:
        val = os.getenv(f"{key}_{index}", os.getenv(key))
        return str(val).lower() in {"1", "true", "yes", "on"}
   
    return {
        "PEOPLE_COUNT": check("PEOPLE_COUNT"),
        "HEAT_MAP": check("HEAT_MAP"),
        "RE_ID": check("RE_ID"),
    }
 
def main():
    load_dotenv()
    logger.info("Starting Camera Data Collection Orchestrator...")
 
    settings = AppSettings()
    api_client = StoreAPIClient(settings)
    s3_client = S3Client()
 
    # Fetch Stores
    store_map = api_client.fetch_store_map()
   
    # Load Configs
    camera_data = load_cameras_from_env()
    if not camera_data:
        logger.warning("No cameras configured. Exiting.")
        return
 
    workers: List[CameraWorker] = []
 
    for idx, data in camera_data.items():
        try:
            config = CameraConfig(**data)
        except TypeError as e:
            logger.error(f"Invalid config for camera index {idx}: {e}")
            continue
 
        features = get_feature_flags(idx)
       
        if not any(features.values()):
            logger.info(f"Camera {config.id}: No features enabled. Skipping.")
            continue
 
        store_name = store_map.get(str(config.store_id), f"store_{config.store_id}")
       
        worker = CameraWorker(
            config=config,
            settings=settings,
            features=features,
            s3_client=s3_client,
            store_folder_name=store_name
        )
        workers.append(worker)
        worker.start()
        logger.info(f"Started worker for Camera {config.id} -> {store_name}")
 
    logger.info(f"Orchestrator running with {len(workers)} cameras.")
 
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=5)
   
    logger.info("Orchestrator stopped.")
 
if __name__ == "__main__":
    main()
 