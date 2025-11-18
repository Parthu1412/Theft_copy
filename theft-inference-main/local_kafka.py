#!/usr/bin/env python3
"""
Video Generation Processor
- Receives raw frame data from Theft Detection Processor via ZeroMQ
- Generates MP4 videos from raw frames
- Uploads videos to S3 and sends messages to RabbitMQ/Kafka
- Similar to custom_process.py but uses PyZMQ instead of Redis
"""

import os
import sys
import time
import zmq
import logging
import asyncio
import uuid
import datetime
import numpy as np
import cv2
import boto3
from botocore.exceptions import ClientError
import subprocess
import concurrent.futures
import signal

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import config
from app.utils.message import TheftMessage

# from app.kafka.asyncio.producer import CustomAIOKafkaProducer
# from app.RMQ.producer import TheftDetectionProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("video_generator")


class VideoGenerationProcessor:
    def __init__(self):
        # ZeroMQ setup
        self.context = zmq.Context()

        # Receiver from Theft Detection Processor
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.connect("tcp://localhost:5559")
        self.receiver.setsockopt(zmq.RCVHWM, 50)

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

        logger.info("Video Generation Processor: Initialized")
        logger.info(
            "Video Generation Processor: Connected to Theft Detection Processor (port 5559)"
        )

    def create_folder(self):
        """Create folder for theft videos"""
        try:
            os.makedirs("theft_videos", exist_ok=True)
            logger.info(
                "Video Generation Processor: Created/verified theft_videos folder"
            )
        except Exception as e:
            logger.error(f"Error creating video folder: {e}")

    def upload_file_and_get_direct_url(self, file_name, bucket, object_name=None):
        """Upload file to S3 and get direct URL"""
        if object_name is None:
            object_name = file_name

        s3 = boto3.client(
            "s3",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name="us-east-2",
        )

        try:
            s3.upload_file(file_name, bucket, object_name)
            url = (
                f"https://{bucket}.s3.{s3.meta.region_name}.amazonaws.com/{object_name}"
            )
            logger.info(
                f"File {file_name} uploaded successfully to {bucket}/{object_name}"
            )
            return url
        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            return None

    def send_rabbitmq_message(
        self, camera_id, url, trace_id, timestamp, theft_res, store_id
    ):
        """Send message to RabbitMQ"""
        try:
            self.rmq_producer.publish_detection(
                trace_id, camera_id, url, timestamp, theft_res, store_id
            )
            logger.info(
                f"Sent theft detection message to RabbitMQ for camera {camera_id} with trace_id {trace_id}"
            )
        except Exception as e:
            logger.error(
                f"Error sending message to RabbitMQ for camera {camera_id}: {e}"
            )
            while True:
                try:
                    self.rmq_producer = TheftDetectionProducer()
                    logger.info("Reconnected to RABBITMQ")
                    break
                except:
                    time.sleep(1)
                    continue
            self.rmq_producer.publish_detection(
                trace_id, camera_id, url, timestamp, theft_res, store_id
            )
            logger.info(
                f"Sent theft detection message to RabbitMQ for camera {camera_id} with trace_id {trace_id}"
            )

    def write_video(self, frames, output_path, fps=15):
        """Write video using FFmpeg"""
        if not frames:
            logger.warning(f"No frames to write for {output_path}")
            return

        try:
            command = [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-r",
                str(fps),
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
                frame = frame_data["frame"]
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

    async def write_video_async(self, frames, output_path, fps=15):
        """Asynchronous video writing method"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self.write_video, frames, output_path, fps
            )
            logger.info(f"Async video processing completed for {output_path}")
        except Exception as e:
            logger.error(f"Error in async video processing: {e}")

    async def process_video_with_cleanup(self, video_data):
        """Process video and handle cleanup"""
        task = asyncio.current_task()
        self.active_tasks.add(task)
        try:
            camera_id = video_data["camera_id"]
            raw_frames = video_data["raw_frames"]
            theft_score = video_data["theft_score"]
            detection_time = video_data["detection_time"]
            frame_count = video_data["frame_count"]

            # Generate filename
            timestamp = datetime.datetime.fromtimestamp(detection_time)
            timestamp_filename = timestamp.strftime("%Y_%m_%dT%H_%M_%S%f")[:-3]
            timestamp_filename = f"{timestamp_filename}_{camera_id}"
            video_path = f"theft_videos/{timestamp_filename}.mp4"

            logger.info(
                f"Processing video for camera {camera_id}: {frame_count} frames -> {video_path}"
            )

            # Calculate FPS (assume 15 FPS default)
            fps = int(os.getenv("VIDEO_FPS", "15"))

            # Process video in background thread
            await self.write_video_async(raw_frames, video_path, fps=fps)

            # Clean up frames data after processing
            del raw_frames

            # Check if video file was created successfully
            if not os.path.exists(video_path):
                logger.error(f"Video file not created: {video_path}")
                return

            # Video processing completed, now handle messaging
            trace_id = str(uuid.uuid4())
            store_id = os.getenv("STORE_ID")

            # Upload to S3
            # url = self.upload_file_and_get_direct_url(
            #     file_name=video_path,
            #     bucket=config.AWS_BUCKET,
            #     object_name=config.AWS_OBJECT_NAME + '/' + video_path
            # )

            # if url:
            #     self.total_videos_uploaded += 1
            #     logger.info(f"Video uploaded to S3: {url}")

            #     # Create message
            #     message = TheftMessage(
            #         camera_id=camera_id,
            #         timestamp=detection_time,
            #         s3_url=url,
            #         trace_id=trace_id,
            #         theft_probability=theft_score,
            #         model_version="v1.0.0"
            #     )

            #     # Send to RabbitMQ
            #     timestamp_str = timestamp.isoformat()
            #     self.send_rabbitmq_message(camera_id, url, trace_id, timestamp_str, theft_score, store_id)

            #     # Send to Kafka
            #     await kafka_producer.produce(
            #         topic=os.getenv("KAFKA_TOPIC", 'theft-detect-topic'),
            #         message=message
            #     )
            #     logger.info(f"Messages sent for camera {camera_id}")
            # else:
            #     logger.error(f"Failed to upload video for camera {camera_id}")

            # # Remove local video file
            # try:
            #     os.remove(video_path)
            #     logger.info(f"Local video file removed: {video_path}")
            # except Exception as e:
            #     logger.warning(f"Could not remove local video file {video_path}: {e}")

            self.total_videos_processed += 1
            logger.info(f"Video processing completed for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error in video processing for camera {camera_id}: {e}")
        finally:
            # Remove task from active tasks
            self.active_tasks.discard(task)

    async def run(self):
        """Main processing loop"""
        logger.info("Video Generation Processor: Starting main loop...")

        # Initialize connections
        # while True:
        #     try:
        #         self.rmq_producer = TheftDetectionProducer()
        #         logger.info("Connected to RABBITMQ")
        #         kafka_producer = CustomAIOKafkaProducer()
        #         await kafka_producer.start()
        #         logger.info("Connected to KAFKA")
        #         break
        #     except Exception as e:
        #         logger.error(f"Error connecting to Kafka or RabbitMQ: {str(e)}")
        #         time.sleep(1)
        #         continue

        try:
            while True:
                try:
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
                        asyncio.create_task(self.process_video_with_cleanup(video_data))

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
                        and self.total_videos_received % 10 == 0
                    ):
                        logger.info(
                            f"Video Generation Stats: Received={self.total_videos_received}, "
                            f"Processed={self.total_videos_processed}, "
                            f"Uploaded={self.total_videos_uploaded}"
                        )

                except Exception as e:
                    logger.error(f"Error in video generation loop: {e}")
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Video Generation Processor: Received interrupt signal")
        except Exception as e:
            logger.error(f"Video Generation Processor: Critical error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Video Generation Processor: Cleaning up...")

        # Close ZeroMQ socket
        if hasattr(self, "receiver"):
            self.receiver.close()
        if hasattr(self, "context"):
            self.context.term()

        # Shutdown executor
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        logger.info("Video Generation Processor: Cleanup completed")

    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


async def main():
    """Main entry point"""
    logger.info("Video Generation Processor: Starting...")

    processor = VideoGenerationProcessor()
    await processor.run()


if __name__ == "__main__":

    def signal_handler(signum, frame):
        logger.info(
            f"Video Generation Processor received signal {signum}, shutting down..."
        )
        exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(main())
