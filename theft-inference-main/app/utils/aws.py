import os
import asyncio
import logging
from aiokafka.abc import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

import boto3
import botocore
from botocore.exceptions import ClientError

from app import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(process)d:%(thread)d] - %(levelname)s: - %(message)s",
)
logger = logging.getLogger("video_generator")


def oauth_cb(oauth_config):
    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token(config.AWS_REGION)

    return auth_token, expiry_ms / 1000


class AWSTokenProvider(AbstractTokenProvider):
    async def token(self):
        return await asyncio.get_running_loop().run_in_executor(None, self._token)

    def _token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token(config.AWS_REGION)
        return token


class S3Client:
    """A simple S3 client to upload and download files."""

    def __init__(self) -> None:
        """Initialize the S3 client with AWS credentials and bucket info."""

        self.bucket = config.AWS_BUCKET
        self.object_name = config.AWS_OBJECT_NAME
        self.aws_access_key_id = config.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = config.AWS_SECRET_ACCESS_KEY

    def upload_file_and_get_direct_url(self, file_name: str, object_name: str = ""):
        """Upload a file to an S3 bucket and get a presigned URL."""

        if not object_name:
            object_name = self.object_name

        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=botocore.config.Config(max_pool_connections=20),
        )

        try:
            s3.upload_file(file_name, self.bucket, object_name)

            # Generate a presigned URL
            # url = s3.generate_presigned_url(
            #     "get_object",
            #     Params={"Bucket": self.bucket, "Key": self.object_name},
            #     ExpiresIn=expiration,
            # )
            url = f"https://{self.bucket}.s3.{s3.meta.region_name}.amazonaws.com/{object_name}"

            logger.info(
                f"File {file_name} uploaded successfully to {self.bucket}/{object_name}"
            )

            return url

        except ClientError as e:
            print(f"Error uploading file: {e}")
            return None

    def download_file(self, file_name: str):

        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=botocore.config.Config(max_pool_connections=20),
        )

        try:
            s3.download_file(self.bucket, self.object_name, file_name)

            logger.info(
                f"File {file_name} downloaded successfully from {self.bucket}/{self.object_name}"
            )
            return file_name

        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            return None
