import os
import requests
import logging
import time
import json
import random
from typing import Dict, Optional

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(operation_func, max_retries=3, operation_name="operation"):
    retry_count = 0
    while retry_count < max_retries:
        try:
            result = operation_func()
            if result:
                return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during {operation_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during {operation_name}: {e}")

        wait_time = random.randint(300, 600)
        logger.info(f"Retrying {operation_name} in {wait_time}s...")
        time.sleep(wait_time)
        retry_count += 1

    logger.error(f"{operation_name} failed after retries")
    return ""


class TokenManager:
    # Singleton instance
    _instance = None

    # Class-level dictionary to store tokens by stream ID
    _stream_tokens: Dict[str, str] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Creating new TokenManager instance")
            cls._instance = super(TokenManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_url=None, email=None, password=None):
        # Initialize only once
        if getattr(self, "_initialized", False):
            return

        self._base_url = base_url or os.getenv("API_BASE_URL")
        self._antmedia_base_url = os.getenv("ANTMEDIA_BASE_URL")
        self._email = email or os.getenv("API_EMAIL")
        self._password = password or os.getenv("API_PASSWORD")
        self._env = os.getenv("API_ENV", "default")
        self._auth_token = None

        # Authenticate immediately if credentials are available
        if self._base_url and self._email and self._password:
            self.get_auth_token()
        else:
            logger.error("Missing API credentials (base_url, email, or password)")

        self._initialized = True

    @property
    def base_url(self):
        return self._base_url

    @property
    def email(self):
        return self._email

    @property
    def password(self):
        return self._password

    @property
    def env(self):
        logger.info(f"Environment: {self._env}")
        return self._env

    @property
    def auth_token(self):
        return self._auth_token

    def is_protech_env(self):
        return self.env.lower() == "protech"

    def _graphql_login(self):
        """Authenticate using GraphQL API"""
        graphql_url = f"{self.base_url}/graphql"

        # GraphQL mutation for login
        query = """
        mutation Login($email: String!, $password: String!) {
            login(email: $email, password: $password) {
                data
                message
            }
        }
        """

        variables = {"email": self.email, "password": self.password}

        payload = {"query": query, "variables": variables}

        logger.info(f"Authenticating to GraphQL API at {graphql_url}")
        response = requests.post(graphql_url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL authentication failed: {data['errors']}")
                return None
            else:
                login_data = data.get("data", {}).get("login", {})
                auth_data = login_data.get("data")
                # Since "data" is a JSON scalar, we need to check if it's a string that needs parsing
                if isinstance(auth_data, str):
                    try:
                        auth_data = json.loads(auth_data)
                    except json.JSONDecodeError:
                        auth_data = {}
                elif not isinstance(auth_data, dict):
                    auth_data = {}

                auth_token = (
                    auth_data.get("token") if isinstance(auth_data, dict) else None
                )
                if auth_token:
                    logger.info("Successfully authenticated via GraphQL")
                    return auth_token
                else:
                    logger.error("Token not found in GraphQL response")
                    return None
        else:
            logger.error(
                f"GraphQL authentication failed: {response.status_code} - {response.text}"
            )
            return None

    def _rest_api_login(self):
        """Authenticate using REST API"""
        login_url = f"{self.base_url}/auth/v2/ml-login"
        login_data = {"email": self.email, "password": self.password}

        logger.info(f"Authenticating to REST API at {login_url}")
        response = requests.post(login_url, json=login_data)

        if response.status_code == 200:
            data = response.json()
            auth_token = data.get("data", {}).get("token")

            if auth_token:
                logger.info("Successfully authenticated via REST API")
                return auth_token
            else:
                logger.error("Token not found in REST response")
                return None
        else:
            logger.error(
                f"REST authentication failed: {response.status_code} - {response.text}"
            )
            return None

    def get_auth_token(self):
        def auth_operation():
            if self.is_protech_env():
                auth_token = self._graphql_login()
            else:
                auth_token = self._rest_api_login()

            if auth_token:
                self._auth_token = auth_token
                return auth_token
            return None

        return retry_with_backoff(auth_operation, operation_name="auth")

    def get_play_token_graphql(self, stream_id):
        graphql_url = f"{self.base_url}/graphql"
        # GraphQL query for getting camera by stream ID
        query = """
        query GetCameraByStreamId($streamId: String!) {
            getCameraByStreamId(streamId: $streamId) {
                id
                schoolId
                camId
                streamUrl
                cameraName
                token
                area
                active
                createdAt
                updatedAt
            }
        }
        """
        variables = {"streamId": stream_id}
        payload = {"query": query, "variables": variables}
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = requests.post(graphql_url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL token retrieval failed: {data['errors']}")
                # Check if the error is due to invalid authentication
                error_message = str(
                    data.get("errors", [{}])[0].get("message", "")
                ).lower()
                if "unauthorized" in error_message or "token" in error_message:
                    logger.info("Auth token expired, refreshing and retrying")
                    self._auth_token = self.get_auth_token()
                    return None
            else:
                token = (
                    data.get("data", {}).get("getCameraByStreamId", {}).get("token", "")
                )
                if token:
                    logger.info(
                        f"Successfully retrieved token for stream {stream_id} via GraphQL"
                    )
                    # Cache the token
                    self._stream_tokens[stream_id] = token
                    return token
                else:
                    logger.warning(
                        f"Token not found in GraphQL response for stream {stream_id}"
                    )
                    return ""
        elif response.status_code == 401:
            logger.info("Auth token expired, refreshing and retrying")
            self._auth_token = self.get_auth_token()
            return None
        else:
            logger.error(
                f"Failed to get token via GraphQL: {response.status_code} - {response.text}"
            )
            return ""

    def get_play_token_rest(self, stream_id):
        url = f"{self._antmedia_base_url}/ms/antmedia/camera/getTokenByStreamId/{stream_id}"
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            token = data.get("data", {}).get("token", "")
            if token:
                logger.info(
                    f"Successfully retrieved token for stream {stream_id} via REST API"
                )
                # Cache the token
                self._stream_tokens[stream_id] = token
                return token
            else:
                logger.warning(
                    f"Token not found in REST response for stream {stream_id}"
                )
                return ""
        elif response.status_code == 401:
            logger.info("Auth token expired, refreshing and retrying")
            self._auth_token = self.get_auth_token()
            return None
        else:
            logger.error(
                f"Failed to get token via REST API: {response.status_code} - {response.text}"
            )
            return ""

    def get_play_token(self, stream_id):
        if stream_id in self._stream_tokens:
            logger.info(f"Using cached token for stream ID: {stream_id}")
            return self._stream_tokens[stream_id]

        def token_operation():
            logger.info(f"Getting token for stream ID: {stream_id}")

            if not self.auth_token:
                self._auth_token = self.get_auth_token()
                if not self.auth_token:
                    logger.error("Failed to get auth token")
                    return ""

            if self.is_protech_env():
                return self.get_play_token_graphql(stream_id)
            else:
                return self.get_play_token_rest(stream_id)

        return retry_with_backoff(
            token_operation, operation_name=f"token retrieval for stream {stream_id}"
        )
