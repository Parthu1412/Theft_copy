import ssl
import pika
import json
from loguru import logger
from datetime import datetime, timezone

from app import config


class RabitMQClient:
    def __init__(self):
        """Initialize the RabbitMQ client with connection parameters."""

        self.connection = self.connect()

        self.channel = self.connection.channel()

    def connect(self):
        """Establish a connection to RabbitMQ server."""

        try:
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = (
                ssl.CERT_NONE
            )  # Warning: This disables SSL certificate verification

            credentials = pika.PlainCredentials(
                config.RABBITMQ_USER_PRODUCER, config.RABBITMQ_PASS_PRODUCER
            )

            # Set up connection parameters
            params = pika.ConnectionParameters(
                host=config.RABBITMQ_HOST_PRODUCER,
                port=config.RABBITMQ_PORT_PRODUCER,
                credentials=credentials,
                ssl_options=pika.SSLOptions(context),
                connection_attempts=3,
                retry_delay=5,
                socket_timeout=10,
            )

            connection = pika.BlockingConnection(params)

            return connection

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise ConnectionError("Failed to connect to RabbitMQ")

    def publish_message(
        self, message: dict, queue_name=config.RABBITMQ_QUEUE_NAME_PRODUCER
    ):

        if self.connection.is_closed:
            self.connection = self.connect()

        self.channel.queue_declare(queue=queue_name, durable=True)

        if config.STORE_MESSAGE_RMQ == "True":  # by default True
            self.channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                ),
            )

        logger.info(f" [x] Sent {message} to queue {queue_name}")

    def close(self):
        self.connection.close()
