import time
import logging
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from app import config

logger = logging.getLogger("TheftInference")

# Configure GPU memory growth
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(config.GPU_LIMIT)]
    )

# Enable XLA and fast matmul/conv on Ampere+ (TF32)
tf.config.optimizer.set_jit(True)
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

# Mixed precision (uses Tensor Cores on supported GPUs)
try:
    from tensorflow import keras as _keras

    _keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    try:
        from tensorflow.keras import mixed_precision as _mixed_precision

        _mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass


@dataclass
class TheftInferenceState:
    """State tracking for each camera's theft detection."""

    camera_id: str
    consecutive_count: int = 0
    theft_probs: List[float] = field(default_factory=list)

    def update_probability(self, prob: float):
        """Update theft probability and manage consecutive count."""

        if prob >= config.THEFT_THRESHOLD:
            self.consecutive_count += 1
            self.theft_probs.append(prob)

            logger.info(
                f"Camera {self.camera_id}: Theft detected! Count: {self.consecutive_count}, Prob: {prob:.4f}"
            )

        else:
            if self.consecutive_count > 0:
                logger.info(
                    f"Camera {self.camera_id}: {prob:.4f} below threshold, "
                    f"resetting count from {self.consecutive_count} to 0"
                )
                self.reset_count()

    def reset_count(self):
        """Reset the consecutive count and stored probabilities."""
        self.consecutive_count = 0
        self.theft_probs.clear()

    def should_trigger_alert(self, current_prob: float) -> Tuple[bool, float]:
        """
        Check if alert should be triggered based on consecutive detections.

        Parameters:
            current_prob: The current theft probability

        Returns:
            Tuple of (trigger_alert, probability_to_use)
        """
        trigger_alert = self.consecutive_count >= config.THEFT_CONT_PREDS

        if trigger_alert:
            # Use max probability from consecutive detections when triggering alert
            prob_to_return = max(self.theft_probs) if self.theft_probs else current_prob
            logger.info(
                f"CONSECUTIVE THEFT DETECTED in camera {self.camera_id} "
                f"for {self.consecutive_count} times with max prob: {prob_to_return:.4f}"
            )
            self.reset_count()
        else:
            prob_to_return = current_prob

        return trigger_alert, prob_to_return


class TheftInferenceService:
    """
    Theft Inference Service using optimized TensorFlow model with state tracking per camera.

    Features:
    - XLA compilation for optimized inference
    - Mixed precision (FP16) for faster computation
    - Per-camera state tracking
    - Automatic model warmup
    """

    def __init__(self):
        """Initialize the TheftInference model and parameters."""

        self.threshold_prob = config.THEFT_THRESHOLD
        self.model_path = config.MODEL_PATH

        if not self.model_path:
            raise ValueError("MODEL_PATH is not set in the .env file")

        # Load the model
        self.model = self.load_model()

        if self.model is None:
            raise ValueError(f"Failed to load the model from {self.model_path}.")

        # Create optimized inference function
        self.infer = self._create_inference_function()

        # Use dataclass-based theft state tracking by camera
        self.theft_state: Dict[str, TheftInferenceState] = {}

        logger.info("TheftInference: Service initialized successfully")

    def load_model(self) -> Optional[tf.Module]:
        """Load the TensorFlow model from the specified path."""

        try:
            # Load using Keras for better optimization support
            if self.model_path.endswith(".h5") or self.model_path.endswith(".keras"):
                model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded Keras model from {self.model_path}")
            else:
                # For SavedModel format
                model = tf.saved_model.load(self.model_path)
                logger.info(f"Loaded SavedModel from {self.model_path}")

            return model

        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            return None

    def _create_inference_function(self):
        """Create an inference function with XLA compilation."""

        def _infer_impl(x):
            return self.model(x, training=False)

        try:
            # Try to create XLA-compiled function
            infer = tf.function(
                _infer_impl,
                jit_compile=True,
                input_signature=[
                    tf.TensorSpec(shape=[1, 100, 224, 224, 3], dtype=tf.float32)
                ],
            )
            logger.info("TheftInference: Created XLA-compiled inference function")
            return infer

        except Exception as e:
            logger.warning(
                f"TheftInference: XLA compilation failed, using regular tf.function: {e}"
            )
            # Fallback to regular tf.function
            infer = tf.function(
                _infer_impl,
                input_signature=[
                    tf.TensorSpec(shape=[1, 100, 224, 224, 3], dtype=tf.float32)
                ],
            )
            return infer

    def get_camera_state(self, camera_id: str) -> TheftInferenceState:
        """Get or create camera state."""

        if camera_id not in self.theft_state:
            self.theft_state[camera_id] = TheftInferenceState(camera_id=camera_id)
            logger.info(f"TheftInference: Created state for camera {camera_id}")

        return self.theft_state[camera_id]

    def predict_with_tensors(
        self, camera_id: str, tensors: List[tf.Tensor]
    ) -> Tuple[bool, float]:
        """
        Optimized predict method using pre-converted tensors.

        Parameters:
            camera_id: Identifier for the camera (to track consecutive predictions)
            tensors: List of pre-converted TensorFlow tensors

        Returns:
            Tuple of (should_trigger, theft_probability)
        """

        camera_state = self.get_camera_state(camera_id)

        try:
            t1 = time.time()
            clip_tensor = tf.expand_dims(tf.stack(tensors), axis=0)
            t2 = time.time()
            logger.info(f"Time taken to stack tensors: {t2 - t1:.4f}s")

            prediction = self.model(clip_tensor)
            t3 = time.time()
            logger.info(f"Time taken to predict: {t3 - t2:.4f}s")
            theft_prob = float(prediction[0][1])
            logger.info(f"Total time taken: {t3 - t1:.4f}s")

            camera_state.update_probability(theft_prob)

            should_trigger, prob_to_use = camera_state.should_trigger_alert(theft_prob)
            return should_trigger, prob_to_use

        except Exception as e:
            logger.error(f"Error during theft prediction with tensors: {e}")
            camera_state.reset_count()
            return False, 0.0

    def warm_up(self):
        """Warm up the model with dummy data for optimal performance."""

        try:
            logger.info("TheftInference: Starting model warmup...")

            dummy_frames = []
            for _ in range(100):
                dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_frame = tf.cast(dummy_frame, dtype=tf.float32)
                dummy_frames.append(dummy_frame)

            clip_tensor = tf.expand_dims(tf.stack(dummy_frames), axis=0)

            start_time = time.time()

            # Run warmup inference
            _ = self.model(clip_tensor)

            warmup_time = time.time() - start_time

            logger.info(f"TheftInference: Model warmed up in {warmup_time:.4f}s")

        except Exception as e:
            logger.error(f"TheftInference: Error during model warmup: {e}")

