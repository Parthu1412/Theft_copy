import cv2
import time
import pika
import json
import base64
import numpy as np
import os
import logging
import traceback
from typing import Tuple, Optional


class RabbitMQ:
    # def __init__(self,
    #              host="172.20.48.178",
    #              port= 5672,
    #              username="admin",
    #              password="YfHqxGBW8495YgiXnHBxKht7xaEo33wV",
    #              queue_name="g_Theft-1",
    #              camera_id="0rq8k4sjpmdf59a"):

    def __init__(
        self,
        host=os.getenv("RABBITMQ_HOST", "127.0.0"),
        port=os.getenv("RABBITMQ_PORT", 5672),
        username=os.getenv("RABBITMQ_USER", "user"),
        password=os.getenv("RABBITMQ_PASS", "password"),
        queue_name=os.getenv("QUEUE_NAME", None),
        buffer_size=1,
        camera_id=os.getenv("RABBITMQ_CAMERAID", None),
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.queue_name = queue_name
        self.camera_id = camera_id
        self.connection = None
        self.channel = None

    def connect(self):
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials,
                heartbeat=300,
                blocked_connection_timeout=300,
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # Declare the queue without lazy mode
            self.channel.queue_declare(
                queue=self.queue_name, durable=True, arguments={"x-queue-mode": "lazy"}
            )
            logging.info(f"Connected to RabbitMQ on {self.host}")
        except Exception as e:
            logging.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logging.info("Connection to RabbitMQ closed")

    def reconnect(self):
        logging.info("Attempting to reconnect to RabbitMQ")
        self.close()
        time.sleep(5)
        self.connect()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        while True:
            try:
                if self.connection is None or self.connection.is_closed:
                    logging.info("Connection closed, reconnecting...")
                    self.reconnect()
                method_frame, _, body = self.channel.basic_get(queue=self.queue_name)
                if method_frame:
                    frame = self.process_message(body)
                    if frame is not None:
                        self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                        # logging.info(f"Message acknowledged for camera {self.camera_id}")
                        return True, frame
                    else:
                        # Changed to not requeue if message doesn't match camera_id
                        self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                        logging.debug(
                            f"Processed message didn't match camera {self.camera_id}"
                        )
                else:
                    logging.debug(f"No message available for camera {self.camera_id}")
                time.sleep(0.1)
            except pika.exceptions.AMQPConnectionError as e:
                logging.error(f"AMQP Connection Error: {e}")
                self.reconnect()
            except Exception as e:
                logging.error(
                    f"Error reading message for camera {self.camera_id}: {str(e)}"
                )
                logging.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(1)

    def process_message(self, body) -> Optional[np.ndarray]:
        try:
            message = json.loads(body)
            if message.get("camera_id") == self.camera_id:
                image_data = base64.b64decode(message["payload"])
                frame = cv2.imdecode(
                    np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    frame = cv2.resize(frame, (640, 480))
                    return frame
                else:
                    logging.error("Failed to decode image data")
                    return None
            else:
                logging.debug(f"Message didn't match camera {self.camera_id}")
                return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
        except KeyError as e:
            logging.error(f"Missing key in message: {e}")
        except Exception as e:
            logging.error(f"Error processing message: {e}")
        return None
