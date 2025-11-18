import logging
import os, ssl
from uuid import uuid4
from aiokafka import AIOKafkaProducer

from app import config
from app.utils.aws import AWSTokenProvider
from app.utils.message import TheftMessage
from app.utils.strings import StringSerializer

# from app.utils.json_schema import JSONSerializer
import json
import asyncio


logger = logging.getLogger("AIOKafkaProducer")


def create_ssl_context():
    _ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    _ssl_context.options |= ssl.OP_NO_SSLv2
    _ssl_context.options |= ssl.OP_NO_SSLv3
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE
    _ssl_context.load_default_certs()

    return _ssl_context


class CustomAIOKafkaProducer:
    def __init__(self) -> None:

        logger.info("aiokafka producer instance has been created.")

        self.producer = AIOKafkaProducer(
            # bootstrap_servers=os.getenv("KAFKA_BROKER_ADDRESS", "localhost:9092"),
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=self.serializer,
            max_request_size=5000000,
            loop=asyncio.get_running_loop(),
            client_id=config.KAFKA_CLIENT_ID,
            security_protocol="SASL_SSL",
            ssl_context=create_ssl_context(),
            sasl_mechanism="OAUTHBEARER",
            sasl_oauth_token_provider=AWSTokenProvider(),
        )

        self.string_serializer = StringSerializer("utf-8")

    def serializer(self, value):
        return json.dumps(value).encode("utf-8")

    async def produce(self, topic: str, message: TheftMessage) -> None:

        try:

            key = self.string_serializer(str(uuid4()))
            value1 = message.to_dict()

            logger.info(f"Message key: {key}, value: {value1}")
            res = await self.producer.send_and_wait(topic=topic, key=key, value=value1)
            logger.info(res)

        except Exception as err:
            logger.error(f"{err}")
            self.producer.stop()
            raise

    async def start(self) -> None:

        logger.info("aiokafka producer is connecting to kafka broker")

        await self.producer.start()

        logger.info("aiokafka producer has connected and started.")

    async def stop(self) -> None:

        await self.producer.stop()

        logger.info("aiokafka producer has stopped.")
