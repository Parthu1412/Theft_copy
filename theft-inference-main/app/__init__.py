import os
import datetime
import logging
from ultralytics import YOLO
import logging.config
from dateutil import tz


tzinfo = tz.gettz(os.environ.get("TIMEZONE", "UTC"))

from dotenv import load_dotenv

load_dotenv()

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "%(asctime)s - [%(process)s] - %(levelname)s: - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": "INFO",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "gunicorn": {"propagate": True},
        "gunicorn.access": {"propagate": True},
        "gunicorn.error": {"propagate": True},
        "uvicorn": {"propagate": True},
        "uvicorn.access": {"propagate": True},
        "uvicorn.error": {"propagate": True},
    },
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

__title__ = "MokSa.ai HeatMap services"
__copyright__ = (
    f"Copyright (c) {datetime.date.today().year} mokSa.ai Limited. All rights reserved."
)
__version__ = "0.1.0"

__uptime__ = datetime.datetime.now().isoformat()

__root_dir__ = os.path.dirname(os.path.abspath(__file__))

_names_with_underscore = [
    "__title__",
    "__version__",
    "__copyright__",
    "__uptime__",
    "__root_dir__",
]


__all__ = [_s for _s in dir() if not _s.startswith("_")]
__all__.extend([_s for _s in _names_with_underscore])
