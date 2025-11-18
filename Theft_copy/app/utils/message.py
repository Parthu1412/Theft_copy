from datetime import datetime
from pydantic import BaseModel


class TheftMessage(BaseModel):

    # camera_ip: str
    camera_id: str
    s3_url: str
    timestamp: str  # Changed from datetime to str to match old format
    trace_id: str
    model_version: str = "v1.0.0"
    theft_probability: float
    store_id: int

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class HeatMapMessage(BaseModel):
    """Message format for heatmap data sent to Kafka"""
    store_id: int = None
    moksa_camera_id: int = None  # Moksa camera ID
    heat_map: str = None  # Base64 encoded compressed heatmap
    timestamp: datetime
    model_version: str = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class AisleMessage(BaseModel):
    """Message format for aisle statistics sent to Kafka"""
    store_id: int = None
    moksa_camera_id: int = None  # Moksa camera ID
    aisle_info: dict = None  # Dictionary containing aisle statistics
    timestamp: datetime
    model_version: str = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

class PeopleCountingMessage(BaseModel):
    # model_config = ConfigDict(protected_namespaces=())

    # camera_ip: str
    camera_id: str
    going_in: int
    going_out: int
    timestamp: datetime
    model_version: str = 'v1.0.0'
    moksa_camera_id: int
    store_id: int

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")