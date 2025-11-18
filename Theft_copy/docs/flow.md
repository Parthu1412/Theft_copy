# Theft Copy — Post-Theft Orchestrator Flow

This document captures what happens after `app/core/orchestrators/theft.py` in the `Theft_copy` project and how data flows through the system.

## Process Startup Order (start_all.sh)
- Theft orchestrator: `python3 -m app.core.orchestrators.theft`
- Optional: Heatmap orchestrator: `python3 -m app.core.orchestrators.heatmap` (when `ENABLE_HEATMAP=true`)
- Optional: People counting orchestrator: `python3 -m app.core.orchestrators.people_count` (when `ENABLE_PEOPLE_COUNTING=true`)
- Wait 10 seconds for warmup
- Multi‑YOLO camera orchestrator: `python3 -m app.core.orchestrators.camera_multi_yolo`
  - Uses `NUM_YOLO_WORKERS` to scale YOLO processes and ZMQ ports

## High-Level Flow
1. Multi‑YOLO camera orchestrator starts N YOLO workers and camera threads, distributing cameras round‑robin across workers.
2. Each YOLO worker (`app/core/services/yolo_inference.py`) runs detection and interpolation, then sends per‑frame messages to Theft via ZMQ.
3. Theft orchestrator (`app/core/orchestrators/theft.py`) buffers per‑camera frames and detections into sliding windows, runs TF model, and evaluates consecutive triggers.
4. On trigger, Theft calls `TheftAPIService.submit_theft_alert(...)` which schedules a 20‑second segment alert (with overlap suppression) for HTTP POST to `THEFT_API_ENDPOINT` when the segment ends.

## Inbound Messaging (YOLO → Theft)
- Transport: ZeroMQ
- Per‑worker theft port: `5590 + worker_id`
  - Theft orchestrator binds `PULL` to each port (`tcp://*:{5590+id}`)
  - YOLO/ModelManager connects `PUSH` to `tcp://localhost:{5590+id}`
- Message (from `ModelManager.send_to_theft_detection`):
  - `{
      camera_id: str,
      frame: np.ndarray,
      frame_number: int,
      timestamp: float,
      detections: List[List[int]],
      store_id: str
    }`

## Theft Orchestrator Trigger Logic
- Buffers frames per camera; converts masked frames to tensors; accumulates 100 frames.
- Model (`TheftInferenceService`) outputs probability; counts consecutive positives.
- If `should_trigger` or already `waiting`, and buffer has enough frames, compute:
  - `end_time` from last frame timestamp
  - `start_time = end_time - 20`
  - Call `TheftAPIService.submit_theft_alert(camera_id, theft_probability, start_time, end_time)`
- Clears the original frame buffer for that camera after submitting.

## TheftAPIService (HTTP Alert Dispatch)
- Overlap suppression: If a previous alert ended at `last_sent_end_time[camera_id]`, shift `start_time = max(start_time, last_sent_end)`.
- Segment sizing: Always sends 20 seconds (`end_time = adjusted_start + 20`).
- Countdown worker: Queues/schedules the alert and sends it exactly when `end_time` is reached (or immediately if already passed).
- Payload (from `build_payload`):
  - `start, end, trace_id, camera_id, store_id, theft_probability, model_version`
  - Optional `url`: built from `config.STREAM_BASE_URL` and the camera’s `url` from `CameraConfigLoader`
  - Optional `moksa_camera_id` from camera config
- Request: `POST` JSON to `config.THEFT_API_ENDPOINT` (`Content-Type: application/json`), timeout `config.API_REQUEST_TIMEOUT`.

## Optional Parallel Branches (from YOLO)
- Heatmap (when enabled):
  - ZMQ `PUSH` on per‑worker port `5509 + worker_id` to Heatmap service at ~1 FPS.
- People counting (when enabled):
  - ZMQ `PUSH` with formatted detections to People Counting service at full FPS.

## Sequence (Mermaid)
```mermaid
sequenceDiagram
  autonumber
  participant Cam as Camera Threads
  participant YOLO as YOLO Workers (N)
  participant Theft as Theft Orchestrator
  participant API as TheftAPIService (HTTP)
  participant Heat as Heatmap (optional)
  participant PC as People Counting (optional)

  Cam->>YOLO: Frames via per‑worker multiprocessing Queue
  YOLO->>Theft: ZMQ PUSH 5590+id {frame, detections, ts, ids}
  par optional
    YOLO->>Heat: ZMQ PUSH 5509+id (1 FPS)
    YOLO->>PC: ZMQ PUSH (full FPS)
  end
  Theft->>Theft: Buffer/window; mask + resize; TF inference
  alt trigger reached
    Theft->>API: submit_theft_alert(start=end-20, end)
    API->>API: Overlap suppression; schedule countdown to end
    API-->>API: At end_time ⇒ HTTP POST to THEFT_API_ENDPOINT
  else no trigger
    Theft->>Theft: Keep buffering; no API call
  end
```

## Key Endpoints and Ports
- ZMQ (YOLO → Theft): `tcp://localhost:5590..(5590+NUM_YOLO_WORKERS-1)`
- ZMQ (YOLO → Heatmap, optional): `tcp://*:(5509 + worker_id)`
- HTTP Alerts: `config.THEFT_API_ENDPOINT`
- Stream URL base: `config.STREAM_BASE_URL` (used if camera websocket/url is available)

## References
- Theft orchestrator: `app/core/orchestrators/theft.py`
- Theft inference: `app/core/services/inference.py`
- API alerts: `app/core/services/api_service.py`
- Model routing: `app/core/services/model_manager.py`
- YOLO worker: `app/core/services/yolo_inference.py`
- Camera configs: `app/utils/camera_config.py`
- Startup script: `start_all.sh`
