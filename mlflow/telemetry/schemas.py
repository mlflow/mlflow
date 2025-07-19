import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from mlflow.version import IS_MLFLOW_SKINNY, IS_TRACING_SDK_ONLY, VERSION


class APIStatus(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class BaseParams:
    """
    Base class for params that are logged to telemetry.
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class LoggedModelParams(BaseParams):
    flavor: str


@dataclass
class RegisteredModelParams(BaseParams):
    is_prompt: bool


@dataclass
class CreateRunParams(BaseParams):
    # Capture the set of currently imported packages at run creation time to
    # understand how MLflow is used together with other libraries. Collecting
    # this data at run creation ensures accuracy and completeness.
    imports: list[str]


@dataclass
class APIRecord:
    api_module: str
    api_name: str
    timestamp_ns: int
    params: Optional[BaseParams] = None
    status: APIStatus = APIStatus.UNKNOWN
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "api_module": self.api_module,
            "api_name": self.api_name,
            # dump params to string so we can parse them easily in ETL pipeline
            "params": self.params.to_json() if self.params else None,
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
    session_id: str = uuid.uuid4().hex
    source_sdk: str = get_source_sdk().value
    mlflow_version: str = VERSION
    schema_version: int = 1
    python_version: str = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    operating_system: str = platform.platform()
    backend_store_scheme: Optional[str] = None
