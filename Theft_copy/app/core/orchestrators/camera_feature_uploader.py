#!/usr/bin/env python3
"""
Camera Feature Uploader
-----------------------
Reads per–camera configuration from environment variables and performs:

1. People Count snapshot upload (if PEOPLE_COUNT flag true)
2. Heat Map snapshot upload (if HEAT_MAP flag true) – currently just a raw frame placeholder
3. Re-ID hourly video chunk recording & upload (if RE_ID flag true)

Environment Variable Conventions (per camera):
  CAMERA_ID_1, CAMERA_URL_1, CLIENT_TYPE_1, STORE_ID_1, WEBSOCKET_URL_1, MOKSA_CAMERA_ID_1 (optional)
Feature flags (either global or per camera):
  PEOPLE_COUNT (global) or PEOPLE_COUNT_1
  HEAT_MAP (global) or HEAT_MAP_1
  RE_ID (global) or RE_ID_1

Additional optional vars:
  TOTAL_CAMERAS (fallback 0 – will auto-detect by scanning CAMERA_ID_i)
  RE_ID_INTERVAL_SECONDS (default 3600) – length of each re-id video segment

S3 Upload Path Pattern:
  theft_videos/store_<store_id>/camera_<camera_id>/<filename>

Where filename examples:
  <timestamp>_<camera_id>_people.jpg
  <timestamp>_<camera_id>_heatmap.jpg
  <startTs>_<endTs>_<camera_id>_reid.mp4

This module is read-only for existing code and can be launched independently:
  python -m app.core.orchestrators.camera_feature_uploader

It reuses CameraInit and S3Client from existing utilities.
"""

import os
import cv2
import time
import threading
import datetime as dt
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from app.config import CameraConfig  # dataclass
from app.utils.camera_intialize import CameraInit
from app.utils.aws import S3Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("camera_feature_uploader")


def _bool_env(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _timestamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")


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
    ) -> None:
        super().__init__(name=f"CamWorker-{config.id}", daemon=True)
        self.cfg = config
        self.people_count_enabled = people_count_enabled
        self.heat_map_enabled = heat_map_enabled
        self.re_id_enabled = re_id_enabled
        self.re_id_interval = re_id_interval
        self.s3 = s3
        self.stop_event = threading.Event()
        self.cap = None
        self.video_writer = None
        self.current_segment_start: Optional[float] = None
        self.frames_written = 0

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
        # Standardize size like other modules
        try:
            frame = cv2.resize(frame, (640, 480))
        except Exception:
            pass
        return frame

    def _ensure_dir(self, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    def _local_base_dir(self) -> str:
        return os.path.join(
            "theft_videos", f"store_{self.cfg.store_id}", f"camera_{self.cfg.id}"
        )

    def _upload_file(self, local_path: str) -> None:
        object_name = f"{os.getenv('AWS_OBJECT_NAME','')}/{local_path}".lstrip("/")
        url = self.s3.upload_file_and_get_direct_url(local_path, object_name)
        if url:
            logger.info(f"Uploaded {local_path} -> {url}")
        else:
            logger.error(f"Upload failed for {local_path}")

    def _snapshot_and_upload(self, frame, tag: str) -> None:
        ts = _timestamp()
        filename = f"{ts}_{self.cfg.id}_{tag}.jpg"
        local_path = os.path.join(self._local_base_dir(), filename)
        self._ensure_dir(local_path)
        try:
            cv2.imwrite(local_path, frame)
            self._upload_file(local_path)
            # Optionally remove local after upload to save space
            try:
                os.remove(local_path)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Camera {self.cfg.id}: Snapshot {tag} failed - {e}")

    def _start_new_segment(self) -> None:
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.current_segment_start = time.time()
        start_str = dt.datetime.utcfromtimestamp(self.current_segment_start).strftime(
            "%Y%m%dT%H%M%S"
        )
        # Placeholder end (will update after closing)
        filename = f"{start_str}_inprogress_{self.cfg.id}_reid.mp4"
        local_path = os.path.join(self._local_base_dir(), filename)
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
        start_str = dt.datetime.utcfromtimestamp(self.current_segment_start).strftime(
            "%Y%m%dT%H%M%S"
        )
        end_str = dt.datetime.utcfromtimestamp(end_time).strftime("%Y%m%dT%H%M%S")
        base_dir = self._local_base_dir()
        # Find the in-progress file
        inprogress = [f for f in os.listdir(base_dir) if f.endswith("_reid.mp4") and f.startswith(start_str)]
        if inprogress:
            old_name = os.path.join(base_dir, inprogress[0])
            new_name = os.path.join(
                base_dir, f"{start_str}_{end_str}_{self.cfg.id}_reid.mp4"
            )
            try:
                os.rename(old_name, new_name)
                self._upload_file(new_name)
                # Optionally remove local file after upload
                try:
                    os.remove(new_name)
                except Exception:
                    pass
                logger.info(
                    f"Camera {self.cfg.id}: Finalized re-id segment {new_name} frames={self.frames_written}"
                )
            except Exception as e:
                logger.error(f"Camera {self.cfg.id}: Rename/upload failed - {e}")
        self.video_writer = None
        self.current_segment_start = None

    def run(self) -> None:
        if not self._open_camera():
            return

        # Capture initial snapshots (one-off) if enabled
        initial_frame = None
        for _ in range(40):  # attempt to grab a valid frame
            frame = self._read_frame()
            if frame is not None:
                initial_frame = frame
                break
            time.sleep(0.1)

        if initial_frame is None:
            logger.warning(f"Camera {self.cfg.id}: No frame available for initial snapshots")
        else:
            if self.people_count_enabled:
                self._snapshot_and_upload(initial_frame, "people")
            if self.heat_map_enabled:
                self._snapshot_and_upload(initial_frame, "heatmap")

        # Start re-id segment if enabled
        if self.re_id_enabled:
            self._start_new_segment()

        # Main loop
        while not self.stop_event.is_set():
            frame = self._read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # Append to current re-id segment
            if self.re_id_enabled and self.video_writer and self.current_segment_start:
                try:
                    self.video_writer.write(frame)
                    self.frames_written += 1
                except Exception as e:
                    logger.error(f"Camera {self.cfg.id}: Video write error - {e}")

                elapsed = time.time() - self.current_segment_start
                if elapsed >= self.re_id_interval:
                    self._finalize_segment()
                    # Start next segment immediately
                    self._start_new_segment()

            # Sleep lightly to reduce CPU usage
            time.sleep(0.01)

        # Shutdown
        if self.re_id_enabled:
            self._finalize_segment()
        if self.cap is not None:
            try:
                if hasattr(self.cap, "release"):
                    self.cap.release()
            except Exception:
                pass
        logger.info(f"Camera {self.cfg.id}: Worker stopped")

    def stop(self):
        self.stop_event.set()


def load_camera_configs() -> Dict[int, Dict[str, Any]]:
    """Discover cameras from env variables sequentially until a gap, or up to TOTAL_CAMERAS."""
    cameras: Dict[int, Dict[str, Any]] = {}
    total = int(os.getenv("TOTAL_CAMERAS", "0") or 0)
    if total <= 0:
        # Auto-detect up to a reasonable max
        max_scan = 50
        for i in range(1, max_scan + 1):
            if os.getenv(f"CAMERA_ID_{i}"):
                total = max(total, i)
        if total == 0:
            logger.warning("No cameras detected via environment variables")
            return cameras

    for i in range(1, total + 1):
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
            "moksa_camera_id": int(os.getenv(f"MOKSA_CAMERA_ID_{i}", "0") or 0),
        }
    logger.info(f"Loaded {len(cameras)} camera configurations")
    return cameras


def build_camera_config(raw: Dict[str, Any]) -> CameraConfig:
    return CameraConfig(
        id=raw["id"],
        index=raw["index"],
        url=raw["url"],
        client_type=raw["client_type"],
        store_id=raw["store_id"],
        websocket_url=raw["websocket_url"],
        moksa_camera_id=raw["moksa_camera_id"],
    )


def feature_flag(flag: str, index: int) -> bool:
    # Per camera flag takes precedence, fallback to global.
    return _bool_env(os.getenv(f"{flag}_{index}", os.getenv(flag)))


def main():
    logger.info("Camera Feature Uploader starting…")
    re_id_interval = int(os.getenv("RE_ID_INTERVAL_SECONDS", "60"))
    cameras = load_camera_configs()
    if not cameras:
        return

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
        worker = CameraFeatureWorker(
            cfg, people, heat, reid, re_id_interval, s3
        )
        worker.start()
        workers.append(worker)
        logger.info(
            f"Camera {cfg.id}: Features -> people={people} heat={heat} re_id={reid} interval={re_id_interval}s"
        )

    logger.info(f"Started {len(workers)} active camera workers")
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C)")
        for w in workers: 
            w.stop()
        for w in workers:
            w.join(timeout=5)
    logger.info("Camera Feature Uploader stopped")


if __name__ == "__main__":
    main()
