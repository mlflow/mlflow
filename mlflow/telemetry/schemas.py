import platform
import sys
import uuid
from dataclasses import dataclass
from enum import Enum

from mlflow.version import VERSION


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


@dataclass
class BaseParams:
    """
    Base class for params that are logged to telemetry.
    """


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
class Record:
    api_name: str
    params: BaseParams | None = None
    status: APIStatus = APIStatus.UNKNOWN.value
    # TODO: add duration_ms after we get approval


@dataclass
class TelemetryInfo:
    session_id: str = uuid.uuid4().hex
    mlflow_version: str = VERSION
    python_version: str = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    operating_system: str = platform.platform()
    backend_store: str | None = None
