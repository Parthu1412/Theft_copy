import uuid
import time
import threading
import logging
import requests
from dataclasses import dataclass
from typing import Dict

from app import config
from app.utils.camera_config import CameraConfigLoader

logger = logging.getLogger("TheftAPIService")


@dataclass
class APIConfiguration:
    """Configuration for theft detection API."""

    api_endpoint: str
    store_id: str
    model_version: str
    request_timeout: float = 0.001  # seconds

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_endpoint:
            raise ValueError("API endpoint cannot be empty")


@dataclass
class PendingAlert:
    """Tracks pending alert for a camera with countdown to end_time."""

    camera_id: str
    start_time: int
    end_time: int
    theft_probability: float
    remaining_time: int  # Seconds until end_time is reached

    def decrement(self) -> None:
        """Decrement remaining time by 1 second."""
        self.remaining_time -= 1

    def is_ready_to_send(self) -> bool:
        """Check if countdown reached 0 and alert is ready to send."""
        return self.remaining_time <= 0


class TheftAPIService:
    """
    Theft API Service for submitting theft detection alerts.

    Features:
    - Fire-and-forget API submission
    - Automatic error handling
    """

    def __init__(self):
        """Initialize the TheftAPIService with configuration."""

        # Load API configuration
        self.config = APIConfiguration(
            api_endpoint=config.THEFT_API_ENDPOINT,
            store_id=config.STORE_ID,
            model_version=config.MODEL_VERSION,
            request_timeout=config.API_REQUEST_TIMEOUT,
        )

        # Load camera configurations for websocket URL lookup
        self.camera_configs = CameraConfigLoader().load_camera_configs()

        # Track pending alerts per camera (countdown mechanism)
        self.pending_alerts: Dict[str, PendingAlert] = {}
        self.pending_alerts_lock = threading.Lock()

        # Track last sent end_time per camera (for overlap detection)
        self.last_sent_end_time: Dict[str, int] = {}

        # Start background countdown thread
        self.stop_event = threading.Event()
        self.countdown_thread = threading.Thread(
            target=self._countdown_worker, daemon=True, name="AlertCountdownWorker"
        )
        self.countdown_thread.start()

        logger.info("TheftAPIService: Initialized with overlap detection")

    def _countdown_worker(self):
        """Background worker that decrements remaining_time and sends alerts when countdown reaches 0."""
        logger.info("TheftAPIService: Countdown worker started")

        while not self.stop_event.is_set():
            time.sleep(1)  # Tick every second

            with self.pending_alerts_lock:
                cameras_to_send = []

                # Decrement remaining_time for all pending alerts
                for camera_id, alert in self.pending_alerts.items():
                    alert.decrement()

                    if alert.is_ready_to_send():
                        cameras_to_send.append(camera_id)

                # Send alerts that reached countdown 0
                for camera_id in cameras_to_send:
                    alert = self.pending_alerts.pop(camera_id)
                    logger.info(
                        f"TheftAPIService: Countdown reached 0 for {camera_id}, "
                        f"sending alert (start: {alert.start_time}, end: {alert.end_time}, duration: {alert.end_time - alert.start_time}s)"
                    )
                    # Send the alert
                    self._send_alert_to_api(
                        camera_id=alert.camera_id,
                        theft_probability=alert.theft_probability,
                        start_time=alert.start_time,
                        end_time=alert.end_time,
                    )
                    # Update last sent end_time for overlap detection
                    self.last_sent_end_time[camera_id] = alert.end_time

        logger.info("TheftAPIService: Countdown worker stopped")

    def build_payload(
        self,
        camera_id: str,
        theft_probability: float,
        start_time: int,
        end_time: int,
        trace_id: str,
    ) -> Dict:
        """
        Build the API request payload.

        Parameters:
            camera_id: Identifier for the camera
            theft_probability: Detected theft probability
            start_time: Video segment start time
            end_time: Video segment end time
            trace_id: Unique trace ID for request tracking

        Returns:
            Dictionary containing the API payload
        """
        # Look up camera-specific configuration first
        websocket_url = None
        moksa_camera_id = 0
        camera_store_id = self.config.store_id  # Default to global store_id
        
        for camera_config in self.camera_configs.values():
            if camera_config.id == camera_id:
                websocket_url = camera_config.websocket_url
                moksa_camera_id = camera_config.moksa_camera_id
                camera_store_id = camera_config.store_id  # Use camera-specific store_id
                break

        payload = {
            "start": start_time,
            "end": end_time,
            "trace_id": trace_id,
            "camera_id": camera_id,
            "store_id": camera_store_id,  # Use camera-specific store_id
            "theft_probability": theft_probability,
            "model_version": self.config.model_version,
        }

        if websocket_url:
            stream_url_to_use = f"{config.STREAM_BASE_URL}/{camera_config.url}.m3u8"
            payload["url"] = stream_url_to_use

        if moksa_camera_id:
            payload["moksa_camera_id"] = moksa_camera_id

        return payload

    def submit_theft_alert(
        self,
        camera_id: str,
        theft_probability: float,
        start_time: int,
        end_time: int,
    ) -> bool:
        """
        Submit or accumulate theft alert with overlap detection and countdown mechanism.

        - Detects overlap with previously sent alert and adjusts start_time
        - Always sends 20 seconds of video (adjusted_start + 20)
        - Calculates countdown based on when end_time will be reached
        - Sends alert when countdown reaches 0 (end_time is reached)

        Parameters:
            camera_id: Identifier for the camera
            theft_probability: Detected theft probability (0.0 to 1.0)
            start_time: Video segment start time (unix timestamp)
            end_time: Video segment end time (unix timestamp - ignored, recalculated)

        Returns:
            True always (accumulation is always successful)
        """

        current_time = int(time.time())

        # Remove overlap with last sent alert
        last_sent_end = self.last_sent_end_time.get(camera_id, 0)
        adjusted_start = max(start_time, last_sent_end)
        overlap_removed = adjusted_start - start_time

        # Always send 20 seconds of video
        end_time = adjusted_start + 20

        # Calculate time until end_time is reached
        remaining_time = end_time - current_time

        # Send immediately if end_time already reached
        if remaining_time <= 0:
            logger.info(
                f"TheftAPIService: Sending immediately for {camera_id} "
                f"(start={adjusted_start}, end={end_time}"
                + (
                    f", removed {overlap_removed}s overlap)"
                    if overlap_removed > 0
                    else ")"
                )
            )
            self._send_alert_to_api(
                camera_id, theft_probability, adjusted_start, end_time
            )
            self.last_sent_end_time[camera_id] = end_time
            return True

        # Otherwise, create or update pending alert
        with self.pending_alerts_lock:
            alert = self.pending_alerts.get(camera_id)
            is_new = alert is None

            if is_new:
                alert = PendingAlert(
                    camera_id,
                    adjusted_start,
                    end_time,
                    theft_probability,
                    remaining_time,
                )
                self.pending_alerts[camera_id] = alert
            else:
                alert.end_time = end_time
                alert.remaining_time = remaining_time

            logger.info(
                f"TheftAPIService: {'Created' if is_new else 'Updated'} pending alert for {camera_id} "
                f"(end={end_time}, remaining={remaining_time}s"
                + (
                    f", removed {overlap_removed}s overlap)"
                    if is_new and overlap_removed > 0
                    else ")"
                )
            )

        return True

    def _send_alert_to_api(
        self,
        camera_id: str,
        theft_probability: float,
        start_time: int,
        end_time: int,
    ) -> bool:
        """
        Internal method to send theft alert to API endpoint (fire-and-forget).

        Parameters:
            camera_id: Identifier for the camera
            theft_probability: Detected theft probability (0.0 to 1.0)
            start_time: Video segment start time (unix timestamp)
            end_time: Video segment end time (unix timestamp)

        Returns:
            True if submission was successful, False otherwise
        """

        try:
            # Generate unique trace ID for request tracking
            trace_id = str(uuid.uuid4())

            # Build payload
            payload = self.build_payload(
                camera_id=camera_id,
                theft_probability=theft_probability,
                start_time=start_time,
                end_time=end_time,
                trace_id=trace_id,
            )

            print(payload)

            # Prepare headers
            headers = {"Content-Type": "application/json"}

            logger.info(
                f"TheftAPIService: Submitting theft alert for {camera_id} "
                f"with probability {theft_probability:.4f} (trace_id: {trace_id})"
            )

            response = requests.post(
                self.config.api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.config.request_timeout,
                stream=True,  # Don't download response body
            )

            # Check response status
            response.raise_for_status()

            logger.info(
                f"TheftAPIService: Theft alert sent successfully for {camera_id}"
            )

            # API call successful
            return True

        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"TheftAPIService: Connection Error - Cannot connect to API endpoint "
                f"{self.config.api_endpoint} for {camera_id}: {e}"
            )
            # API call failed - connection error
            return False

        except requests.exceptions.Timeout as e:
            logger.warning(
                f"TheftAPIService: Timeout Error - API call timed out for {camera_id} "
                f"after {self.config.request_timeout}s: {e}"
            )
            # API call failed - timeout error
            return False

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"TheftAPIService: HTTP Error - API returned error for {camera_id}: {e}"
            )
            return False

        except Exception as e:
            logger.error(
                f"TheftAPIService: Unexpected error sending theft alert for {camera_id}: {e}"
            )
            return False

    def cleanup(self):
        """Cleanup resources and stop countdown worker."""
        logger.info("TheftAPIService: Stopping countdown worker...")
        self.stop_event.set()
        if self.countdown_thread.is_alive():
            self.countdown_thread.join(timeout=2.0)
        logger.info("TheftAPIService: Cleanup completed")

