#!/usr/bin/env python3
"""
Video Generation Orchestrator
- Receives raw frame data from Theft Detection Processor via ZeroMQ
- Generates MP4 videos from raw frames
- Uploads videos to S3 and sends messages to RabbitMQ/Kafka
- Similar to custom_process.py but uses PyZMQ instead of Redis
"""

import logging
import asyncio
import cv2, zmq
import numpy as np
import concurrent.futures
import datetime, subprocess
import os, sys, signal, uuid
from dataclasses import dataclass

# Load environment variables from .env file 
from dotenv import load_dotenv

load_dotenv()

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import config
from app.utils.aws import S3Client
from app.utils.message import TheftMessage
from app.mqtt.rabitmq import RabitMQClient
from app.kafka.asyncio.producer import CustomAIOKafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("video_generator")


class VideoGenerationOrchestrator:
    def __init__(self):
        # ZeroMQ setup
        self.context = zmq.Context()

        # Receiver from Theft Detection Processor
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect("tcp://localhost:5559")
        self.receiver.setsockopt(zmq.RCVHWM, 50)

        self.s3 = S3Client()

        # Create folder for videos
        self.create_folder()

        # Create multi-thread executor for parallel video processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Track active video processing tasks
        self.active_tasks = set()

        # Statistics
        self.total_videos_received = 0
        self.total_videos_processed = 0
        self.total_videos_uploaded = 0

        logger.info("Video Generation Orchestrator: Initialized")
        logger.info(
            "Video Generation Orchestrator: Connected to Theft Detection Processor (port 5559)"
        )

    def create_folder(self):
        """Create folder for theft videos"""

        try:

            os.makedirs("theft_videos", exist_ok=True)
            logger.info(
                "Video Generation Orchestrator: Created/verified theft_videos folder"
            )

        except Exception as e:
            logger.error(f"Error creating video folder: {e}")

    def write_video(self, frames: list[np.ndarray], output_path: str):
        """Write video using FFmpeg"""

        if not frames:
            logger.warning(f"No frames to write for {output_path}")
            return

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            command = [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-r",
                "15",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "35",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-tune",
                "zerolatency",  # Low latency
                output_path,
            ]

            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )

            for frame_data in frames:
                if frame_data is None:
                    continue

                # Convert numpy array to JPEG bytes
                frame = frame_data
                if isinstance(frame, np.ndarray):
                    _, buffer = cv2.imencode(".jpg", frame)
                    process.stdin.write(buffer.tobytes())

            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                stderr_output = process.stderr.read().decode()
                logger.error(f"FFmpeg error for {output_path}: {stderr_output}")
            else:
                logger.info(f"Video written successfully: {output_path}")

        except Exception as e:
            logger.error(f"Error writing video {output_path}: {e}")

    async def write_video_async(self, frames: list[np.ndarray], output_path: str):
        """Asynchronous video writing method"""

        try:

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self.write_video, frames, output_path
            )
            logger.info(f"Async video processing completed for {output_path}")

        except Exception as e:
            logger.error(f"Error in async video processing: {e}")

    async def process_video_with_cleanup(
        self, video_data: dict, kafka_producer: CustomAIOKafkaProducer
    ):
        """Process video and handle cleanup"""

        try:
            camera_id = video_data["camera_id"]
            frames = video_data["frames"]
            theft_score = video_data["theft_score"]
            detection_time = video_data["timestamp"]
            frame_count = video_data["frame_count"]
            store_id = video_data.get("store_id", "unknown")

            # Generate filename - handle timestamp with 'Z' suffix
            timestamp = datetime.datetime.fromisoformat(detection_time[:-1])  # Remove 'Z'
            timestamp_filename = timestamp.strftime("%Y_%m_%dT%H_%M_%S%f")[:-3]
            timestamp_filename = f"{timestamp_filename}_{camera_id}"
            video_path = f"theft_videos/store_{store_id}/camera_{camera_id}/{timestamp_filename}.mp4"

            logger.info(
                f"Processing video for camera {camera_id}: {frame_count} frames -> {video_path}"
            )

            # Process video in background thread
            await self.write_video_async(frames, video_path)

            # Clean up frames data after processing
            del frames

            # Check if video file was created successfully
            if not os.path.exists(video_path):
                logger.error(f"Video file not created: {video_path}")
                return

            # Video processing completed, now handle messaging
            trace_id = str(uuid.uuid4())

            # Upload to S3
            url = self.s3.upload_file_and_get_direct_url(
                file_name=video_path,
                object_name=config.AWS_OBJECT_NAME + "/" + video_path,
            )

            if url:
                self.total_videos_uploaded += 1
                logger.info(f"Video uploaded to S3: {url}")

                # Create message
                message = TheftMessage(
                    camera_id=camera_id,
                    timestamp=detection_time[:-1] if detection_time.endswith('Z') else detection_time,  # Remove 'Z' if present
                    s3_url=url,
                    trace_id=trace_id,
                    theft_probability=theft_score,
                    model_version="v1.0.0",
                    store_id=store_id,
                )

                # Send to Kafka
                await kafka_producer.produce(message=message, topic=config.KAFKA_TOPIC)
                logger.info(
                    f"Kafka Message sent for camera {message} to Kafka topic {config.KAFKA_TOPIC}"
                )

                # Send to RabbitMQ
                queue_name = f"theft_{store_id}"
                try:
                    self.rmq_client.publish_message(
                        message=message.to_dict(),
                        queue_name=queue_name,
                    )
                    logger.info(
                        f"RabbitMQ Message sent for camera {camera_id} to RabbitMQ queue "
                        f"{queue_name} with trace_id {trace_id}"
                    )
                except Exception as e:
                    logger.error(f"RabbitMQ publish failed for camera {camera_id}: {e}")
                    while True:
                        try:
                            self.rmq_client = RabitMQClient()
                            self.rmq_client.publish_message(
                                message=message.to_dict(),
                                queue_name=queue_name,
                            )
                            logger.info(
                                f"RabbitMQ Message sent for camera {camera_id} to RabbitMQ queue "
                                f"{queue_name} with trace_id {trace_id}"
                            )
                            break
                        except Exception as e:
                            logger.error(f"RabbitMQ publish failed for camera {camera_id}: {e}")
                            continue

            else:
                logger.error(f"Failed to upload video for camera {camera_id}")

            # # Remove local video file
            try:
                os.remove(video_path)
                logger.info(f"Local video file removed: {video_path}")
            except Exception as e:
                logger.warning(f"Could not remove local video file {video_path}: {e}")

            self.total_videos_processed += 1
            logger.info(f"Video processing completed for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error in video processing for camera {camera_id}: {e}")
        finally:
            # Remove task from active tasks
            self.active_tasks.discard(asyncio.current_task().get_name())

    async def run(self):
        """Main processing loop"""

        logger.info("Video Generation Orchestrator: Starting main loop...")

        # Initialize connections
        try:

            self.rmq_client = RabitMQClient()
            logger.info("Connected to RABBITMQ")

        except ConnectionError as e:
            logger.error(f"Error connecting to RabbitMQ: {str(e)}")
            return

        try:

            kafka_producer = CustomAIOKafkaProducer()
            await kafka_producer.start()
            logger.info("Connected to KAFKA")

        except Exception as e:
            logger.error(f"Error connecting to Kafka or RabbitMQ: {str(e)}")
            return

        try:
            while True:
                # Check for new video data (non-blocking)
                try:
                    video_data = self.receiver.recv_pyobj(zmq.NOBLOCK)

                    self.total_videos_received += 1

                    logger.info(
                        f"Received video data for camera {video_data['camera_id']}: "
                        f"{video_data['frame_count']} frames"
                    )

                    # Process all videos in parallel (no blocking)
                    # Start async video processing (non-blocking)
                    task = asyncio.create_task(
                        self.process_video_with_cleanup(video_data, kafka_producer)
                    )
                    self.active_tasks.add(task.get_name())

                    logger.info(
                        f"Started async video processing for camera {video_data['camera_id']}"
                    )

                except zmq.Again:
                    # No video data available
                    pass

                # Fast polling since we're not blocking
                await asyncio.sleep(0.1)

                # Periodic status logging
                if (
                    self.total_videos_received > 0
                    and self.total_videos_received % 50 == 0
                ):
                    logger.info(
                        f"Video Generation Stats: Received={self.total_videos_received}, "
                        f"Processed={self.total_videos_processed}, "
                        f"Uploaded={self.total_videos_uploaded}"
                    )

        except KeyboardInterrupt:
            logger.info("Video Generation Orchestrator: Received interrupt signal")

        except Exception as e:
            logger.error(f"Video Generation Orchestrator: Critical error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Video Generation Orchestrator: Cleaning up...")

        # Close ZeroMQ socket
        if hasattr(self, "receiver"):
            self.receiver.close()

        if hasattr(self, "context"):
            self.context.term()

        # Shutdown executor
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        logger.info("Video Generation Orchestrator: Cleanup completed")

    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


async def main():
    """Main entry point"""
    logger.info("Video Generation Orchestrator: Starting...")

    processor = VideoGenerationOrchestrator()
    await processor.run()


if __name__ == "__main__":

    def signal_handler(signum, frame):
        logger.info(
            f"Video Generation Orchestrator received signal {signum}, shutting down..."
        )
        exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(main())
