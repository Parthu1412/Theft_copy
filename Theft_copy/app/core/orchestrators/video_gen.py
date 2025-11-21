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
from typing import List
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
            os.makedirs("theft_vidoes_temp4", exist_ok=True)
            logger.info("Video Generation Orchestrator: Created/verified theft_videos folder")
        except Exception as e:
            logger.error(f"Error creating video folder: {e}")

    def write_video(self, frames: List[np.ndarray], output_path: str):
        # Check if we actually received frames
        if not frames:
            logger.warning(f"No frames to write for {output_path}")
            return

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            #FFmpeg command setup
            command = [
                "ffmpeg",
                "-y",
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "-r", "15",
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "35",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-tune", "zerolatency",
                output_path,
            ]

            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )

            count = 0
            for i, frame in enumerate(frames):
                if frame is None:
                    continue
                
              
                if isinstance(frame, list):
                    frame = np.array(frame, dtype=np.uint8)

                # Ensure it is now a valid array before encoding
                if isinstance(frame, np.ndarray):
                    try:
                        ok, buffer = cv2.imencode(".jpg", frame)
                        if ok:
                            process.stdin.write(buffer.tobytes())
                            count += 1
                        else:
                            logger.warning(f"Frame {i} failed to encode")
                    except IOError:
                        break
                else:
                    logger.warning(f"Frame {i} is invalid type: {type(frame)}")

            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                stderr_output = process.stderr.read().decode()
                logger.error(f"FFmpeg error for {output_path}: {stderr_output}")
            elif count == 0:
                logger.error(f"FFmpeg ran but wrote 0 frames to {output_path}. Check input data.")
            else:
                logger.info(f"Video written successfully: {output_path} ({count} frames)")

        except Exception as e:
            logger.error(f"Error writing video {output_path}: {e}")

    async def write_video_async(self, frames: List[np.ndarray], output_path: str):
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
            video_path = os.path.join("theft_vidoes_temp4", f"store_{store_id}", f"camera_{camera_id}", filename)

            logger.info(
                f"Processing video for camera {camera_id}: {frame_count} frames -> {video_path}"
            )

            # Process video in background thread
            await self.write_video_async(frames, video_path)
            
            del frames

            # Check if video file was created successfully
            if not os.path.exists(video_path) or os.path.getsize(video_path) < 100:
                logger.error(f"Video file validation failed (missing or empty): {video_path}")
                return
            
            trace_id = str(uuid.uuid4())

            # Upload to S3
            url = self.s3.upload_file_and_get_direct_url(
                file_name=video_path,
                object_name=config.AWS_OBJECT_NAME + "/" + video_path,
            )

            if url:
                self.total_videos_uploaded += 1
                logger.info(f"Video uploaded to S3: {url}")

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
                
                # Send to RabbitMQ (Safe Retry Loop)
                queue_name = f"theft_{store_id}"
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if not hasattr(self, 'rmq_client'):
                             self.rmq_client = RabitMQClient()

                        self.rmq_client.publish_message(
                            message=message.to_dict(),
                            queue_name=queue_name,
                        )
                        logger.info(f"RabbitMQ Message sent for {camera_id} to queue {queue_name}")
                        break
                    except Exception as e:
                        logger.error(f"RabbitMQ retry {attempt+1} failed: {e}")
                        await asyncio.sleep(1)
                        try:
                            self.rmq_client = RabitMQClient()
                        except:
                            pass
                else:
                    logger.error(f"Failed to send RabbitMQ message for {camera_id} after retries")
                
                # Clean up local video file (From Code 2)
                # try:
                #     os.remove(video_path)
                #     logger.info(f"Local video file removed: {video_path}")
                # except Exception as e:
                #     logger.warning(f"Could not remove local video file {video_path}: {e}")

            else:
                logger.error(f"Failed to upload video for camera {camera_id}")

            self.total_videos_processed += 1
            logger.info(f"Video processing completed for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error in video processing for camera {video_data.get('camera_id', 'unknown')}: {e}")
        finally:
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

                    # Checking frames empty?
                    frame_count = video_data.get('frame_count', 0)
                    frames_list = video_data.get('frames', [])
                    
                    if not frames_list or frame_count == 0:
                        logger.warning(f"Received EMPTY video payload for {video_data.get('camera_id')}")
                    else:
                        logger.info(f"Received video payload: {frame_count} frames. First frame type: {type(frames_list[0])}")
                        
                    task = asyncio.create_task(self.process_video_with_cleanup(video_data, kafka_producer))
                    self.active_tasks.add(task.get_name())
                except zmq.Again:
                    pass

                await asyncio.sleep(0.1)

                if self.total_videos_received > 0 and self.total_videos_received % 50 == 0:
                    logger.info(f"Video Generation Stats: Processed={self.total_videos_processed}")
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
        logger.info(f"Video Generation Orchestrator received signal {signum}, shutting down...")
        exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(main())