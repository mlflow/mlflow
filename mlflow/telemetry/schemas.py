import json
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mlflow.version import IS_MLFLOW_SKINNY, IS_TRACING_SDK_ONLY, VERSION


class Status(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Record:
    event_name: str
    timestamp_ns: int
    params: dict[str, Any] | None = None
    status: Status = Status.UNKNOWN
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "event_name": self.event_name,
            # dump params to string so we can parse them easily in ETL pipeline
            "params": json.dumps(self.params) if self.params else None,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
        }


class SourceSDK(str, Enum):
    MLFLOW_TRACING = "mlflow-tracing"
    MLFLOW = "mlflow"
    MLFLOW_SKINNY = "mlflow-skinny"


def get_source_sdk() -> SourceSDK:
    if IS_TRACING_SDK_ONLY:
        return SourceSDK.MLFLOW_TRACING
    elif IS_MLFLOW_SKINNY:
        return SourceSDK.MLFLOW_SKINNY
    else:
        return SourceSDK.MLFLOW


@dataclass
class TelemetryInfo:
    session_id: str
    source_sdk: str = get_source_sdk().value
    mlflow_version: str = VERSION
    schema_version: int = 1
    python_version: str = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    operating_system: str = platform.platform()
    tracking_uri_scheme: str | None = None


@dataclass
class TelemetryConfig:
    ingestion_url: str
    disable_events: set[str]
