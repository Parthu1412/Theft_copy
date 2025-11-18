# Theft Inference (Main) — Video Generation Flow

This document summarizes what happens after `video_gen` starts in the `theft-inference-main` project, and how data flows end-to-end.

## Process Startup Order
- Theft orchestrator: `python -m app.core.orchestrators.theft`
- Video generation: `python -m app.core.orchestrators.video_gen`
- Camera orchestrator: `python -m app.core.orchestrators.camera`

## High-Level Flow
1. Camera orchestrator starts camera threads and a YOLO process.
2. YOLO emits frames with detections into ZMQ (Theft orchestrator listens on `tcp://localhost:5558`).
3. Theft orchestrator buffers per-camera frames, runs TF model, and on trigger pushes `video_data` to ZMQ `tcp://localhost:5559`.
4. `video_gen` receives `video_data`, writes an MP4 via ffmpeg, uploads to S3, publishes messages to Kafka and RabbitMQ, and deletes the local MP4.

## Data Contracts
- theft -> video_gen (ZMQ 5559):
  - `{
      camera_id: str,
      frames: List[np.ndarray],
      theft_score: float,
      timestamp: ISO8601 str (e.g., "2024-10-01T12:34:56.789Z"),
      frame_count: int,
      store_id: str
    }`

- Kafka `TheftMessage` (topic: `config.KAFKA_TOPIC`):
  - `camera_id, timestamp, s3_url, trace_id, theft_probability, model_version, store_id`

- RabbitMQ queue name: `theft_{store_id}` with message `TheftMessage.to_dict()`

## Sequence (Mermaid)
```mermaid
sequenceDiagram
  autonumber
  participant Cam as Camera Threads
  participant YOLO as YOLO Process
  participant Theft as Theft Orchestrator
  participant VideoGen as Video Generation
  participant S3 as Amazon S3
  participant Kafka as Kafka
  participant RMQ as RabbitMQ

  Cam->>YOLO: Frames via multiprocessing Queue
  YOLO->>Theft: ZMQ PUSH 5558 {frame, detections, timestamp, camera_id, store_id}
  Theft->>Theft: Buffer/window; TF model inference (consecutive trigger)
  alt trigger reached
    Theft-->>VideoGen: ZMQ PUSH 5559 video_data (frames, score, ts, ids)
    VideoGen->>VideoGen: ffmpeg write MP4 (15fps, libx264, CRF 35)
    VideoGen->>S3: Upload object (prefix `config.AWS_OBJECT_NAME`)
    S3-->>VideoGen: Direct URL
    VideoGen->>Kafka: Publish TheftMessage to `config.KAFKA_TOPIC`
    VideoGen->>RMQ: Publish to queue `theft_{store_id}`
    VideoGen->>VideoGen: Delete local MP4; update stats
  else no trigger
    Theft->>Theft: Keep buffering; no downstream action
  end
```

## Key Endpoints and Ports
- ZMQ (theft receive): `tcp://localhost:5558`
- ZMQ (vidceive): `tcp://localhost:5559`
- Kafka topic: `configeo_gen re.KAFKA_TOPIC`
- RabbitMQ queue: `theft_{store_id}`
- S3 bucket/path: derived from `config.AWS_OBJECT_NAME`

## References
- Theft orchestrator: `app/core/orchestrators/theft.py`
- Video generation: `app/core/orchestrators/video_gen.py`
- Camera orchestrator: `app/core/orchestrators/camera.py`
- Message: `app/utils/message.py`
- AWS client: `app/utils/aws.py` 
- Kafka producer: `app/kafka/asyncio/producer.py`
- RabbitMQ client: `app/mqtt/rabitmq.py`
