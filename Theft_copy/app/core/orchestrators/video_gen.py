#!/usr/bin/env python3
"""
Video Generation Orchestrator (Theft_copy)
- Receives raw frame data from Theft Detection Processor via ZeroMQ
- Generates MP4 videos from raw frames using ffmpeg
- Uploads videos to S3 and sends messages to RabbitMQ/Kafka

Note: The theft orchestrator CONNECTS to tcp://localhost:5559,
so this orchestrator BINDS to tcp://*:5559.
"""

import logging
import asyncio
import cv2, zmq
import numpy as np
import concurrent.futures 
import datetime, subprocess
import os, sys, signal, uuid
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import config
from app.utils.aws import S3Client
from app.utils.message import TheftMessage
from app.mqtt.rabitmq import RabitMQClient
from app.kafka.asyncio.producer import CustomAIOKafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("video_generator")


class VideoGenerationOrchestrator:
    def __init__(self):
        self.context = zmq.Context()

        # Receiver from Theft Detection Processor
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("tcp://*:5559")
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

        logger.info("Video Generation Orchestrator: Initialized (bound on 5559)")

    def create_folder(self):
        try:
            os.makedirs("theft_videos", exist_ok=True)
            logger.info("Video Generation Orchestrator: Created/verified theft_videos folder")
        except Exception as e:
            logger.error(f"Error creating video folder: {e}")

    def write_video(self, frames: list[np.ndarray], output_path: str):
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
                "zerolatency",
                output_path,
            ]

            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )

            for frame in frames:
                if frame is None:
                    continue
                # Convert numpy array to JPEG bytes
                if isinstance(frame, np.ndarray):
                    ok, buffer = cv2.imencode(".jpg", frame)
                    if ok:
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
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.write_video, frames, output_path)
            logger.info(f"Async video processing completed for {output_path}")
        except Exception as e:
            logger.error(f"Error in async video processing: {e}")

    def _normalize_timestamp_for_filename(self, detection_time):
        try:
            if isinstance(detection_time, str):
                ts = detection_time[:-1] if detection_time.endswith('Z') else detection_time
                dt = datetime.datetime.fromisoformat(ts)
            else:
                dt = datetime.datetime.fromtimestamp(float(detection_time))
            # Format with milliseconds
            return dt.strftime("%Y_%m_%dT%H_%M_%S%f")[:-3]
        except Exception:
            # Fallback to now
            return datetime.datetime.utcnow().strftime("%Y_%m_%dT%H_%M_%S%f")[:-3]

    async def process_video_with_cleanup(self, video_data: dict, kafka_producer: CustomAIOKafkaProducer):
        try:
            camera_id = video_data.get("camera_id", "unknown")
            frames = video_data.get("frames", [])
            theft_score = float(video_data.get("theft_score", 0.0))
            detection_time = video_data.get("timestamp")
            frame_count = int(video_data.get("frame_count", len(frames)))
            store_id = video_data.get("store_id", "unknown")

            ts_name = self._normalize_timestamp_for_filename(detection_time)
            ts_for_msg = (
                detection_time[:-1] if isinstance(detection_time, str) and detection_time.endswith('Z') else detection_time
            )
            filename = f"{ts_name}_{camera_id}.mp4"
            video_path = os.path.join("theft_videos", f"store_{store_id}", f"camera_{camera_id}", filename)

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
                    timestamp=ts_for_msg,
                    s3_url=url,
                    trace_id=trace_id,
                    theft_probability=theft_score,
                    model_version=getattr(config, "MODEL_VERSION", "v1.0.0"),
                    store_id=store_id,
                )

                 # Send to Kafka
                await kafka_producer.produce(message=message, topic=config.KAFKA_TOPIC)
                logger.info(
                    f"Kafka Message sent for camera {camera_id} to topic {config.KAFKA_TOPIC}"
                )
                # Send to RabbitMQ
                queue_name = f"theft_{store_id}"
                try:
                    self.rmq_client.publish_message(
                        message=message.to_dict(),
                        queue_name=queue_name,
                    )
                    logger.info(
                        f"RabbitMQ Message sent for camera {camera_id} to queue {queue_name}"
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
                                f"RabbitMQ Message sent for camera {camera_id} to queue {queue_name}"
                            )
                            break
                        except Exception as e:
                            logger.error(f"RabbitMQ publish retry failed for camera {camera_id}: {e}")
                            continue
            else:
                logger.error(f"Failed to upload video for camera {camera_id}")
            # Clean up local video file
            try:
                os.remove(video_path)
                logger.info(f"Local video file removed: {video_path}")
            except Exception as e:
                logger.warning(f"Could not remove local video file {video_path}: {e}")
            
            self.total_videos_processed += 1
            logger.info(f"Video processing completed for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error in video processing for camera {video_data.get('camera_id', 'unknown')}: {e}")
        finally:
            # Remove task from active tasks
            self.active_tasks.discard(asyncio.current_task().get_name())

    async def run(self):
        logger.info("Video Generation Orchestrator: Starting main loop...")

        try:
            self.rmq_client = RabitMQClient()
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Error connecting to RabbitMQ: {e}")
            return

        try:
            kafka_producer = CustomAIOKafkaProducer()
            await kafka_producer.start()
            logger.info("Connected to Kafka")
        except Exception as e:
            logger.error(f"Error connecting to Kafka: {e}")
            return

        try:
            while True:
                try:
                    video_data = self.receiver.recv_pyobj(zmq.NOBLOCK)
                    self.total_videos_received += 1

                    logger.info(
                        f"Received video data for camera {video_data.get('camera_id','?')}: "
                        f"{video_data.get('frame_count', 0)} frames"
                    )

                    task = asyncio.create_task(self.process_video_with_cleanup(video_data, kafka_producer))
                    self.active_tasks.add(task.get_name())
                except zmq.Again:
                    pass

                await asyncio.sleep(0.1)

                if self.total_videos_received > 0 and self.total_videos_received % 50 == 0:
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
        logger.info("Video Generation Orchestrator: Cleaning up...")
        if hasattr(self, "receiver"):
            self.receiver.close()
        if hasattr(self, "context"):
            self.context.term()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        logger.info("Video Generation Orchestrator: Cleanup completed")


async def main():
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
