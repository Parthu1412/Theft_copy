import os
import asyncio
import threading
import time
import logging
import numpy as np
from typing import Optional, Deque
from collections import deque
import ssl
import aiohttp
from aiortc import RTCPeerConnection, RTCIceServer, RTCConfiguration
from antmedia_service.token_api import TokenManager
from antmedia_service.antmedia_client import AntMediaClient
import json

logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("libav.swscaler").setLevel(logging.ERROR)
import av

av.logging.set_level(av.logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AntMediaCamera")
logging.getLogger("aiortc").setLevel(logging.DEBUG)

# STUN server for WebRTC connections
STUN_SERVER_URL = "stun:stun1.l.google.com:19302"


class AntMediaCamera:
    """
    A camera-like interface for AntMedia WebRTC streams that mimics OpenCV's VideoCapture
    Implementation using aiortc library for improved WebRTC compatibility
    """

    # token_manager = TokenManager()

    def __init__(
        self, websocket_url: str, stream_id: str, buffer_size: int = 60
    ) -> None:
        """
        Initialize the AntMediaCamera.

        Args:
            websocket_url: WebSocket URL for the AntMedia server
            stream_id: Stream ID to connect to
            buffer_size: Maximum size of the frame buffer
        """
        self.websocket_url: str = websocket_url
        self.stream_id: str = stream_id

        # Use deque for better performance with maxlen to automatically drop oldest frames
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.buffer_lock: threading.Lock = threading.Lock()  # For thread safety

        self.latest_frame: Optional[np.ndarray] = None
        self.running: bool = True
        self.connected: bool = False

        # Create instance logger with stream ID for better tracking
        self.logger: logging.Logger = logging.getLogger(f"AntMediaCamera-{stream_id}")

        # Get tokens using shared TokenManager - this will use cached token if available
        self.play_token = self._get_play_token()

        # For async/sync bridging
        self.loop = None
        self.client = None
        self.websocket = None
        self.peer_connection = None
        self.stop_event = threading.Event()
        self.connection_established = threading.Event()

        # 1-second buffer for FPS sampling
        self.frame_buffer_1s = []  # Buffer to collect frames for 1 second
        self.buffer_start_time = time.time()
        self.target_fps = 15  # Target FPS after sampling

        # Frame counting for logging
        self.frames_received_count = 0
        self.frames_sampled_count = 0
        self.last_log_time = time.time()

        # Start WebRTC connection in a background thread with event loop
        self.thread: threading.Thread = threading.Thread(
            target=self._connect_and_receive, daemon=True
        )
        self.thread.start()

        # Wait for the connection to be established (with timeout)
        timeout: int = (
            15  # seconds - increased timeout for aiortc which may take longer
        )
        start_time: float = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self.connected:
            self.logger.warning("Connection to AntMedia server timed out")

        self.frame_retry = time.time()

    def _get_play_token(self) -> str:
        """Get the play token for the stream"""
        try:
            # This will use the cached token if it exists
            return None 
            token = AntMediaCamera.token_manager.get_play_token(self.stream_id)
            if token:
                self.logger.info(f"Got play token for stream ID: {self.stream_id}")
                return token
            else:
                self.logger.warning(
                    f"Failed to get play token for stream ID: {self.stream_id}"
                )
                return ""
        except Exception as e:
            self.logger.error(f"Error getting play token: {e}")
            return ""

    def _create_event_loop(self):
        """Create a new event loop for the current thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def _connect_and_receive(self) -> None:
        """Connect to AntMedia server and set up the WebRTC connection in a background thread"""
        try:
            self.logger.info(
                f"Starting aiortc WebRTC connection for stream ID: {self.stream_id}"
            )

            # Create new event loop for this thread
            self.loop = self._create_event_loop()

            # Run the async connection method in the event loop
            self.loop.run_until_complete(self._async_connect_and_receive())

        except Exception as e:
            self.logger.error(f"WebRTC connection error: {e}", exc_info=True)
            self.running = False

    async def _connect_websocket(self):
        """Connect to the AntMedia server WebSocket endpoint"""
        # Directly use the websocket URL from the environment or the provided value
        ws_url = os.getenv("ANTMEDIA_WEBSOCKET_URL", self.websocket_url)
        try:
            self.logger.info(f"Connecting to WebSocket: {ws_url}")
            session = aiohttp.ClientSession()

            # Create SSL context that doesn't verify certificate (for testing)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self.websocket = await session.ws_connect(ws_url, ssl=ssl_context)

            self.logger.info("WebSocket connected")
            self.client = AntMediaClient(
                self.stream_id, self.play_token, self.websocket
            )
            return True
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            return False

    async def _async_connect_and_receive(self):
        """Asynchronous method to connect to the server and handle frames"""
        try:
            # Connect WebSocket
            websocket_connected = await self._connect_websocket()
            if not websocket_connected:
                return

            # Configure WebRTC
            config = RTCConfiguration([RTCIceServer(urls=STUN_SERVER_URL)])
            self.peer_connection = RTCPeerConnection(config)

            # Frame callback to store frames in our buffer
            @self.peer_connection.on("track")
            async def on_track(track):
                self.logger.info(f"Track received: {track.kind}")

                if track.kind == "video":
                    self.logger.info("Video track established")
                    self.connected = True
                    self.connection_established.set()

                    # Process frames
                    while self.running and not self.stop_event.is_set():
                        try:
                            # Get frame from track
                            frame = await track.recv()

                            # Convert frame to numpy array
                            if hasattr(frame, "to_ndarray"):
                                # Convert frame to BGR (OpenCV format)
                                img = frame.to_ndarray(format="bgr24")

                                # Add frame to 1-second buffer with timestamp
                                current_time = time.time()
                                self.frame_buffer_1s.append(
                                    {"frame": img, "timestamp": current_time}
                                )
                                self.frames_received_count += 1

                                # Process buffer every second
                                if current_time - self.buffer_start_time >= 1.0:
                                    # Sample frames to target FPS
                                    sampled_frames = self._sample_frames_to_target_fps(
                                        self.frame_buffer_1s
                                    )

                                    # Add sampled frames to main buffer
                                    with self.buffer_lock:
                                        for frame_data in sampled_frames:
                                            self.frame_buffer.append(
                                                frame_data["frame"]
                                            )
                                            self.latest_frame = frame_data["frame"]

                                    self.frames_sampled_count += len(sampled_frames)

                                    # Log statistics
                                    time_elapsed = current_time - self.last_log_time
                                    if time_elapsed >= 1.0:
                                        received_fps = (
                                            self.frames_received_count / time_elapsed
                                        )
                                        sampled_fps = (
                                            self.frames_sampled_count / time_elapsed
                                        )
                                        self.logger.info(
                                            f"Stream {self.stream_id}: "
                                            f"Received: {received_fps:.1f} FPS ({self.frames_received_count} frames), "
                                            f"Sampled: {sampled_fps:.1f} FPS ({self.frames_sampled_count} frames), "
                                            f"Target: {self.target_fps} FPS"
                                        )

                                        # Reset counters
                                        self.frames_received_count = 0
                                        self.frames_sampled_count = 0
                                        self.last_log_time = current_time

                                    # Reset 1-second buffer
                                    self.frame_buffer_1s = []
                                    self.buffer_start_time = current_time
                        except Exception as e:
                            if not self.stop_event.is_set():
                                self.logger.error(f"Error processing video frame: {e}")
                                await asyncio.sleep(
                                    0.1
                                )  # Don't spin too fast on errors
                elif track.kind == "audio":
                    self.logger.info(
                        "Audio track received - draining to prevent memory leak"
                    )

                    # Drain audio frames to prevent memory leak
                    while self.running and not self.stop_event.is_set():
                        try:
                            # Consume audio frame to prevent accumulation
                            audio_frame = await track.recv()
                            # Discard the audio frame immediately
                            del audio_frame
                        except Exception as e:
                            if not self.stop_event.is_set():
                                self.logger.debug(f"Error processing audio frame: {e}")
                                await asyncio.sleep(
                                    0.1
                                )  # Don't spin too fast on errors
                            else:
                                break

            # Send play request
            await self.client.play(self.peer_connection)

            # Main loop to keep the connection alive
            ping_interval = 30  # seconds
            last_ping_time = time.time()

            # Send pings to keep the connection alive
            while self.running and not self.stop_event.is_set():
                # Check if it's time to send a ping
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    if self.websocket and not self.websocket.closed:
                        try:
                            await self.websocket.send_str('{"command": "ping"}')
                            self.logger.debug("Ping sent")
                            last_ping_time = current_time
                        except Exception as e:
                            self.logger.error(f"Error sending ping: {e}")
                            break

                # Process messages from the WebSocket
                if self.websocket and not self.websocket.closed:
                    try:
                        message = await asyncio.wait_for(
                            self.websocket.receive(), timeout=0.1
                        )
                        if message.type == aiohttp.WSMsgType.TEXT:
                            await self.client.process_message(
                                message.data, self.peer_connection
                            )
                        elif message.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            self.logger.error(f"WebSocket closed: {message}")
                            break
                    except asyncio.TimeoutError:
                        # This is expected, just continue
                        pass
                    except Exception as e:
                        if not self.stop_event.is_set():
                            self.logger.error(
                                f"Error processing WebSocket message: {e}"
                            )

                await asyncio.sleep(0.01)  # Avoid spinning too fast

        except Exception as e:
            self.logger.error(f"Error in async connection handler: {e}", exc_info=True)
        finally:
            # Cleanup on exit
            if hasattr(self, "websocket") and self.websocket:
                if not self.websocket.closed:
                    await self.websocket.close()

            if hasattr(self, "peer_connection") and self.peer_connection:
                await self.peer_connection.close()

            if hasattr(self, "client") and self.client:
                await self.client.close()

            self.logger.info("WebRTC connection closed")

    def _sample_frames_to_target_fps(self, frames_with_timestamps):
        """
        Sample frames from 1-second buffer to achieve target FPS
        Uses uniform sampling to select the best frames
        """
        if not frames_with_timestamps:
            return []

        total_frames = len(frames_with_timestamps)

        # If we have fewer frames than target, return all
        if total_frames <= self.target_fps:
            return frames_with_timestamps

        # Calculate sampling interval
        sampling_interval = total_frames / self.target_fps

        # Sample frames uniformly
        sampled_frames = []
        for i in range(self.target_fps):
            frame_index = int(i * sampling_interval)
            if frame_index < total_frames:
                sampled_frames.append(frames_with_timestamps[frame_index])

        return sampled_frames

    def read(self):
        """
        Read a frame from the WebRTC stream (mimics OpenCV's VideoCapture.read())

        Returns:
            tuple: (success, frame) where success is True if a frame was retrieved
        """
        if not self.connected:
            return False, None

        with self.buffer_lock:
            if len(self.frame_buffer) > 0:
                self.frame_retry = time.time()
                frame = (
                    self.frame_buffer.popleft().copy()
                )  # Get and remove the oldest frame
                return True, frame

        if time.time() - self.frame_retry < 20:
            return False, np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            return False, None

    def release(self) -> None:
        """
        Release resources (mimics OpenCV's VideoCapture.release())
        """
        self.logger.info(
            "Releasing AntMediaCamera resources for stream: %s", self.stream_id
        )

        # Set running to False to stop the loop
        self.running = False
        self.stop_event.set()

        # If we have an event loop, schedule the cleanup
        if self.loop:
            try:
                # Stop the event loop
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception as e:
                self.logger.error(f"Error stopping event loop: {e}")

        # Wait for the background thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.logger.warning(
                    "Background thread did not terminate, may cause resource leaks"
                )

        # Clear references
        self.latest_frame = None
        with self.buffer_lock:
            self.frame_buffer.clear()

        self.logger.info("AntMediaCamera resources released")

    def stop(self):
        """
        Stop method for compatibility with other camera interfaces
        Delegates to release() method
        """
        self.logger.info("Stop method called for AntMediaCamera - calling release()")
        self.release()
