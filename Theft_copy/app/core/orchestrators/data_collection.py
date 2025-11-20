#!/usr/bin/env python3
"""
Camera Data Collection Orchestrator
-----------------------
Reads per–camera configuration from environment variables and performs:
1. People Count snapshot upload
2. Heat Map snapshot upload
3. Re-ID hourly video chunk recording & upload

Folder Structure:
  Data_Collection/<STORE_NAME>/camera_<id>/<model_name>/<filename>

It dynamically fetches the Store Name from the API using the Store ID.
It respects SAVE_IN_S3 and SAVE_TO_LOCAL flags.
"""

import os
import cv2
import time
import threading
import datetime as dt
import logging
import requests
import re
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from app.config import CameraConfig
from app.utils.camera_intialize import CameraInit
from app.utils.aws import S3Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("DataCollectionOrchestrator")

API_LOGIN_URL = os.getenv("API_LOGIN_URL", "http://alpha.api.moksa.ai/auth/login")
API_STORE_URL = os.getenv("API_STORE_URL", "http://alpha.api.moksa.ai/store/getAllStoresForDropdown")

def _bool_env(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}

def _timestamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

def sanitize_filename(name: str) -> str:
    """Replaces spaces and special chars with underscores for safe folder names."""
    return re.sub(r'[^a-zA-Z0-9-]', '_', str(name))

def fetch_store_name_map() -> Dict[str, str]:
    """
    Logs in and fetches all stores. Returns a dict.
    """
    email = os.getenv("API_EMAIL")
    password = os.getenv("API_PASSWORD")
    
    if not email or not password:
        logger.warning("API credentials not found in .env. Using Store IDs.")
        return {}

    try:
        #Login
        logger.info(f"Logging in to API as {email}...")
        auth_resp = requests.post(API_LOGIN_URL, json={"email": email, "password": password}, timeout=10)
        auth_resp.raise_for_status()
        
        # Extract Token
        auth_data = auth_resp.json()
        token = auth_data.get("token") or auth_data.get("accessToken")
        if not token and "data" in auth_data and isinstance(auth_data["data"], dict):
             token = auth_data["data"].get("token")
             
        if not token:
            logger.error(f"Login successful but no token found: {auth_data.keys()}")
            return {}

        # Fetch Stores
        logger.info("Fetching store list from API...")
        headers = {"Authorization": f"Bearer {token}"}
        store_resp = requests.get(API_STORE_URL, headers=headers, timeout=10)
        store_resp.raise_for_status()
        
        json_response = store_resp.json()
        
        stores_list = []
        
        # Logic to handle nested JSON structure from API
        if isinstance(json_response, list):
            stores_list = json_response
        elif isinstance(json_response, dict):
            if "data" in json_response and isinstance(json_response["data"], list):
                stores_list = json_response["data"]
            elif "data" in json_response and isinstance(json_response["data"], dict):
                 inner_data = json_response["data"]
                 if "data" in inner_data and isinstance(inner_data["data"], list):
                     stores_list = inner_data["data"]
            elif "stores" in json_response:
                 stores_list = json_response["stores"]

        if not stores_list:
            logger.warning(f"Could not find store list in API response.")
            return {}

        # Build Map
        store_map = {}
        for store in stores_list:
            if not isinstance(store, dict): continue
            
            s_id = str(store.get("id"))
            s_name = store.get("name", f"Unknown_{s_id}")
            store_map[s_id] = sanitize_filename(s_name)
            
        logger.info(f"Successfully cached names for {len(store_map)} stores.")
        return store_map

    except Exception as e:
        logger.error(f"Failed to fetch store names from API: {e}")
        return {}

class CameraFeatureWorker(threading.Thread):
    """Thread per camera handling snapshots + optional hourly re-id video chunks."""

    def __init__(
        self,
        config: CameraConfig,
        people_count_enabled: bool,
        heat_map_enabled: bool,
        re_id_enabled: bool,
        re_id_interval: int,
        s3: S3Client,
        store_folder_name: str,
    ) -> None:
        super().__init__(name=f"CamWorker-{config.id}", daemon=True)
        self.cfg = config
        self.people_count_enabled = people_count_enabled
        self.heat_map_enabled = heat_map_enabled
        self.re_id_enabled = re_id_enabled
        self.re_id_interval = re_id_interval
        self.s3 = s3
        self.store_folder_name = store_folder_name 
        self.stop_event = threading.Event()
        self.cap = None
        self.video_writer = None
        self.current_segment_start: Optional[float] = None
        self.frames_written = 0

    # Checking flags
    def _should_save_s3(self) -> bool:
        return _bool_env(os.getenv("SAVE_IN_S3", "true"))

    def _should_save_local(self) -> bool:
        return _bool_env(os.getenv("SAVE_TO_LOCAL", "true"))

    def _open_camera(self) -> bool:
        try:
            cam_init = CameraInit(self.cfg)
            self.cap = cam_init.camera_init()
            if self.cap is None:
                raise RuntimeError("Camera init returned None")
            logger.info(f"Camera {self.cfg.id}: Opened source {self.cfg.url}")
            return True
        except Exception as e:
            logger.error(f"Camera {self.cfg.id}: Failed to open - {e}")
            return False

    def _read_frame(self) -> Optional[Any]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        try:
            frame = cv2.resize(frame, (640, 480))
        except Exception:
            pass
        return frame

    def _ensure_dir(self, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    def _local_base_dir(self, model_name: str) -> str:
        return os.path.join(
            "Data_Collection", 
            self.store_folder_name, 
            f"camera_{self.cfg.id}",
            model_name
        )

    def _upload_file(self, local_path: str) -> bool:
        """
        Uploads to S3 if enabled. Returns True if uploaded, False otherwise.
        """
        if not self._should_save_s3():
            return False

        # Clean up S3 path to remove the local root folder name
        root_folder = "Data_Collection"
        if root_folder in local_path:
            relative_path = os.path.relpath(local_path, start=root_folder)
        else:
            relative_path = os.path.basename(local_path)

        aws_prefix = os.getenv('AWS_OBJECT_NAME', '')
        object_name = f"{aws_prefix}/{relative_path}".replace("//", "/").lstrip("/")
        
        url = self.s3.upload_file_and_get_direct_url(local_path, object_name)
        if url:
            logger.info(f"Uploaded {local_path} -> {url}")
            return True
        else:
            logger.error(f"Upload failed for {local_path}")
            return False

    def _handle_file_persistence(self, local_path: str):
        """
        Decides whether to upload and whether to delete locally based on flags.
        """
        # Upload if enabled
        uploaded = self._upload_file(local_path)

        # Delete local if SAVE_TO_LOCAL is false
        if not self._should_save_local():
            try:
                os.remove(local_path)
                # logger.info(f"Deleted local file: {local_path}")
            except Exception:
                pass
        else:
            if not uploaded and self._should_save_s3():
                logger.warning(f"Saved locally (S3 failed): {local_path}")
            else:
                logger.info(f"Saved locally: {local_path}")

    def _snapshot_and_upload(self, frame, tag: str) -> None:
        ts = _timestamp()
        model_folder = "people_count" if tag == "people" else "heat_map"
        filename = f"{ts}_{self.cfg.id}_{tag}.jpg"
        local_path = os.path.join(self._local_base_dir(model_folder), filename)
        
        self._ensure_dir(local_path)
        try:
            # Always write to disk first
            cv2.imwrite(local_path, frame)
            # Handle logic
            self._handle_file_persistence(local_path)
        except Exception as e:
            logger.error(f"Camera {self.cfg.id}: Snapshot {tag} failed - {e}")

    def _start_new_segment(self) -> None:
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.current_segment_start = time.time()
        start_str = dt.datetime.utcfromtimestamp(self.current_segment_start).strftime("%Y%m%dT%H%M%S")
        
        filename = f"{start_str}_inprogress_{self.cfg.id}_reid.mp4"
        local_path = os.path.join(self._local_base_dir("re_id"), filename)
        
        self._ensure_dir(local_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(local_path, fourcc, 15.0, (640, 480))
        self.frames_written = 0
        logger.info(f"Camera {self.cfg.id}: Started new re-id segment {filename}")

    def _finalize_segment(self) -> None:
        if not self.video_writer or self.current_segment_start is None:
            return
        self.video_writer.release()
        end_time = time.time()
        start_str = dt.datetime.utcfromtimestamp(self.current_segment_start).strftime("%Y%m%dT%H%M%S")
        end_str = dt.datetime.utcfromtimestamp(end_time).strftime("%Y%m%dT%H%M%S")
        
        base_dir = self._local_base_dir("re_id")
        if not os.path.exists(base_dir): return

        inprogress = [f for f in os.listdir(base_dir) if f.endswith("_reid.mp4") and f.startswith(start_str)]
        if inprogress:
            old_name = os.path.join(base_dir, inprogress[0])
            new_name = os.path.join(base_dir, f"{start_str}_{end_str}_{self.cfg.id}_reid.mp4")
            try:
                os.rename(old_name, new_name)
                # Handle logic
                self._handle_file_persistence(new_name)
            except Exception as e:
                logger.error(f"Rename/upload failed: {e}")
        self.video_writer = None
        self.current_segment_start = None

    def run(self) -> None:
        if not self._open_camera():
            return

        # Force create folders so they are visible immediately
        if self.people_count_enabled:
            self._ensure_dir(os.path.join(self._local_base_dir("people_count"), "placeholder"))
        if self.heat_map_enabled:
            self._ensure_dir(os.path.join(self._local_base_dir("heat_map"), "placeholder"))
        if self.re_id_enabled:
            self._ensure_dir(os.path.join(self._local_base_dir("re_id"), "placeholder"))

        # Wait for camera warm-up
        initial_frame = None
        logger.info(f"Camera {self.cfg.id}: Waiting for first frame...")
        for i in range(200):  
            frame = self._read_frame()
            if frame is not None:
                initial_frame = frame
                logger.info(f"Camera {self.cfg.id}: First frame captured!")
                break
            time.sleep(0.1)

        if initial_frame is None:
            logger.warning(f"Camera {self.cfg.id}: Timed out waiting for initial frame")
        else:
            if self.people_count_enabled:
                self._snapshot_and_upload(initial_frame, "people")
            if self.heat_map_enabled:
                self._snapshot_and_upload(initial_frame, "heatmap")

        if self.re_id_enabled:
            self._start_new_segment()

        frame_failures = 0
        while not self.stop_event.is_set():
            frame = self._read_frame()
            if frame is None:
                frame_failures += 1
                if frame_failures % 100 == 0:
                    logger.warning(f"Camera {self.cfg.id}: reading None frames (Count: {frame_failures})")
                time.sleep(0.05)
                continue
            
            frame_failures = 0 

            if self.re_id_enabled and self.video_writer and self.current_segment_start:
                try:
                    self.video_writer.write(frame)
                    self.frames_written += 1
                except Exception as e:
                    logger.error(f"Camera {self.cfg.id}: Video write error - {e}")

                elapsed = time.time() - self.current_segment_start
                if elapsed >= self.re_id_interval:
                    self._finalize_segment()
                    self._start_new_segment()
            time.sleep(0.01)

        if self.re_id_enabled:
            self._finalize_segment()
        if self.cap is not None:
            try:
                if hasattr(self.cap, "release"): self.cap.release()
            except Exception: pass
        logger.info(f"Camera {self.cfg.id}: Worker stopped")

    def stop(self):
        self.stop_event.set()


def load_camera_configs() -> Dict[int, Dict[str, Any]]:
    cameras: Dict[int, Dict[str, Any]] = {}
    total = int(os.getenv("TOTAL_CAMERAS", "0") or 0)
    if total <= 0:
        max_scan = 50
        for i in range(1, max_scan + 1):
            if os.getenv(f"CAMERA_ID_{i}"):
                total = max(total, i)
    
    for i in range(1, total + 1):
        cam_id = os.getenv(f"CAMERA_ID_{i}")
        url = os.getenv(f"CAMERA_URL_{i}")
        if not cam_id or not url: continue
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

def build_camera_config(raw: Dict[str, Any]) -> CameraConfig:
    return CameraConfig(
        id=raw["id"], index=raw["index"], url=raw["url"],
        client_type=raw["client_type"], store_id=raw["store_id"],
        websocket_url=raw["websocket_url"], moksa_camera_id=raw["moksa_camera_id"],
    )

def feature_flag(flag: str, index: int) -> bool:
    return _bool_env(os.getenv(f"{flag}_{index}", os.getenv(flag)))

def main():
    logger.info("Camera Feature Uploader starting…")
    
    store_map = fetch_store_name_map()
    re_id_interval = int(os.getenv("RE_ID_INTERVAL_SECONDS", "60"))
    cameras = load_camera_configs()
    
    if not cameras: return

    s3 = S3Client()
    workers = []
    
    for idx, data in cameras.items():
        cfg = build_camera_config(data)
        people = feature_flag("PEOPLE_COUNT", idx)
        heat = feature_flag("HEAT_MAP", idx)
        reid = feature_flag("RE_ID", idx)
        
        if not any([people, heat, reid]):
            logger.info(f"Camera {cfg.id}: No features enabled, skipping")
            continue
        
        # Resolve Store Name
        s_id = str(cfg.store_id)
        store_name = store_map.get(s_id, f"store_{s_id}")
        
        worker = CameraFeatureWorker(
            cfg, people, heat, reid, re_id_interval, s3, store_name
        )
        worker.start()
        workers.append(worker)
        logger.info(f"Started worker for {cfg.id} (Target Folder: {store_name})")

    logger.info(f"Started {len(workers)} active camera workers")
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        for w in workers: w.stop()
        for w in workers: w.join(timeout=5)
    logger.info("Stopped")

if __name__ == "__main__":
    main()