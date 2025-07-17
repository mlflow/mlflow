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


class ModelType(str, Enum):
    MODEL_PATH = "model_path"
    MODEL_OBJECT = "model_object"
    PYTHON_FUNCTION = "python_function"
    PYTHON_MODEL = "python_model"
    CHAT_MODEL = "chat_model"
    CHAT_AGENT = "chat_agent"
    RESPONSES_AGENT = "responses_agent"
    # pyfunc log_model can accept either python_model or loader_module,
    # we set model type to LOADER_MODULE if loader_module is specified
    LOADER_MODULE = "loader_module"


@dataclass
class BaseParams:
    """
    Base class for params that are logged to telemetry.
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class LogModelParams(BaseParams):
    flavor: str
    model: ModelType
    is_pip_requirements_set: bool = False
    is_extra_pip_requirements_set: bool = False
    is_code_paths_set: bool = False
    is_params_set: bool = False
    is_metadata_set: bool = False


@dataclass
class AutologParams(BaseParams):
    flavor: str
    disable: bool
    log_traces: bool
    log_models: bool


@dataclass
class GenaiEvaluateParams(BaseParams):
    scorers: list[str]
    is_predict_fn_set: bool = False


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
