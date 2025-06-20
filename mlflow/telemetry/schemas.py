import platform
import sys
import uuid
from enum import Enum
from typing import Optional

import pydantic
from pydantic import Field

from mlflow.version import VERSION


class APIStatus(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"


class ModelType(str, Enum):
    MODEL_PATH = "model_path"
    MODEL_OBJECT = "model_object"
    PYTHON_MODEL = "python_model"
    CHAT_MODEL = "chat_model"
    CHAT_AGENT = "chat_agent"
    RESPONSES_AGENT = "responses_agent"


class LogModelParams(pydantic.BaseModel):
    flavor: str
    model: ModelType
    pip_requirements: bool = False
    extra_pip_requirements: bool = False
    code_paths: bool = False
    params: bool = False
    metadata: bool = False

    model_config = {"use_enum_values": True}


class AutologParams(pydantic.BaseModel):
    flavor: str
    disable: bool
    log_traces: bool
    log_models: bool


class GenaiEvaluateParams(pydantic.BaseModel):
    scorers: list[str]
    predict_fn: bool = False


class Record(pydantic.BaseModel):
    api_name: str
    params: Optional[dict[str, bool | str]] = None
    status: APIStatus = Field(default=APIStatus.UNKNOWN.value)
    # TODO: add duration_ms after we get approval

    model_config = {"use_enum_values": True}


class TelemetryInfo(pydantic.BaseModel):
    session_id: str = uuid.uuid4().hex
    mlflow_version: str = VERSION
    python_version: str = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    operating_system: str = platform.platform()
    backend_store: str | None = None
