import time
import numpy as np
import tensorflow as tf
from loguru import logger
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app import config
from app.core.data_models.latency import LatencyStats

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(config.GPU_LIMIT)]
    )


@dataclass
class TheftInferenceState:
    """State tracking for each camera's theft detection."""

    camera_id: str
    consecutive_count: int = 0
    theft_probs: list = field(default_factory=list)

    def update_probability(self, prob: float):
        """Update theft probability (if needed for future use)."""

        if prob >= config.THEFT_THRESHOLD:
            self.consecutive_count += 1
            self.theft_probs.append(prob)

            logger.info(
                f"Camera {self.camera_id}: Theft detected! Count: {self.consecutive_count}"
            )

        else:
            if self.consecutive_count > 0:
                logger.info(
                    f"Camera {self.camera_id}: {prob} below threshold, resetting count from {self.consecutive_count} to 0"
                )
                self.reset_count()

    def reset_count(self):
        """Reset the consecutive count."""
        self.consecutive_count = 0
        self.theft_probs.clear()

    def should_trigger_alert(self, current_prob: float) -> tuple[bool, float]:
        """Check if alert should be triggered."""
        trigger_alert = self.consecutive_count >= config.THEFT_CONT_PREDS
        
        if trigger_alert:
            # Use max probability from consecutive detections when triggering alert
            prob_to_return = max(self.theft_probs) if self.theft_probs else current_prob
            logger.info(
                f"CONSECUTIVE THEFT DETECTED in camera {self.camera_id} for {self.consecutive_count} times"
            )
            self.reset_count()
        else:
            # Use current probability when not triggering alert
            prob_to_return = current_prob

        return trigger_alert, prob_to_return


class TheftInferenceService:
    """Theft Inference using TensorFlow model with state tracking per camera."""

    def __init__(self):
        """Initialize the TheftInference model and parameters."""

        self.threshold_prob = config.THEFT_THRESHOLD

        self.model = self.load_model()

        # Use dataclass-based theft state tracking by camera
        self.theft_state: Dict[str, TheftInferenceState] = {}

        # Theft latency stats
        self.latency_stats = LatencyStats()

    def load_model(self):
        """Load the TensorFlow model from the specified path."""

        model_path = config.MODEL_PATH

        if not model_path:
            raise ValueError("MODEL_PATH is not set in the .env file")

        try:
            model = tf.saved_model.load(model_path)

            if model is None:
                raise ValueError(f"Failed to load the model from {model_path}.")

            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def get_camera_state(self, camera_id: str) -> TheftInferenceState:
        """Get or create camera state."""

        if camera_id not in self.theft_state:
            self.theft_state[camera_id] = TheftInferenceState(camera_id=camera_id)

        return self.theft_state[camera_id]

    def predict_with_tensors(
        self, camera_id: str, tensors: List[tf.Tensor]
    ) -> Tuple[bool, float]:
        """
        Optimized predict method using pre-converted tensors

        Parameters:
            - tensor_list: List of pre-converted TensorFlow tensors
            - camera_id: Identifier for the camera (to track consecutive predictions)

        Returns:
            - should_trigger: Boolean indicating if theft alert should be triggered
            - theft_probability: Theft probability value
        """

        camera_state = self.get_camera_state(camera_id)

        try:
            clip_tensor = tf.expand_dims(tf.stack(tensors), axis=0)

            self.latency_stats.total += 1

            st = time.time()

            prediction = self.model(clip_tensor)
            theft_prob = float(prediction[0][1])

            latency = time.time() - st
            self.latency_stats.add_latency(latency)

            camera_state.update_probability(theft_prob)

            # Log stats periodically
            if (
                self.latency_stats.inference_count > 0
                and self.latency_stats.inference_count % config.YOLO_STATS_INTERVAL == 0
            ):
                logger.info(f"Theft Latency Stats: {self.latency_stats}")
                self.latency_stats.reset_latency_stats()

            should_trigger, prob_to_use = camera_state.should_trigger_alert(theft_prob)
            return should_trigger, prob_to_use

        except Exception as e:
            logger.error(f"Error during theft prediction with tensors: {e}")
            camera_state.reset_count()
            return False, 0.0

    def warm_up(self):
        """Warm up the model with dummy data."""

        try:
            dummy_frames = []
            for _ in range(100):

                dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

                dummy_frame = tf.cast(dummy_frame, dtype=tf.float32)

                dummy_frames.append(dummy_frame)

            clip_tensor = tf.expand_dims(tf.stack(dummy_frames), axis=0)

            start_time = time.time()

            _ = self.model(clip_tensor)

            warmup_time = time.time() - start_time

            logger.info(f"Model warmed up in {warmup_time:.4f}s")

        except Exception as e:
            logger.error(f"Error during model warmup: {e}")
