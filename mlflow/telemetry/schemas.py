import json
import platform
import sys
import uuid
from abc import abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from mlflow.telemetry.constant import PACKAGES_TO_CHECK_IMPORT
from mlflow.version import IS_MLFLOW_SKINNY, IS_TRACING_SDK_ONLY, VERSION


class Status(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"


class EventName(str, Enum):
    CREATE_EXPERIMENT = "create_experiment"
    CREATE_RUN = "create_run"
    CREATE_LOGGED_MODEL = "create_logged_model"
    CREATE_REGISTERED_MODEL = "create_registered_model"
    CREATE_MODEL_VERSION = "create_model_version"
    CREATE_PROMPT = "create_prompt"
    START_TRACE = "start_trace"
    LOG_ASSESSMENT = "log_assessment"
    EVALUATE = "evaluate"

    def __str__(self) -> str:
        return self.value


@dataclass
class BaseParams:
    """
    Base class for params that are logged to telemetry.
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    @abstractmethod
    def parse(cls, arguments: dict[str, Any]) -> Optional["BaseParams"]:
        """
        Parse the arguments and return a BaseParams object.
        """


@dataclass
class LoggedModelParams(BaseParams):
    flavor: str

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> Optional["LoggedModelParams"]:
        if flavor := arguments.get("flavor"):
            return LoggedModelParams(flavor=flavor.removeprefix("mlflow."))
        return None


def _is_prompt(tags: dict[str, str]) -> bool:
    try:
        from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
    except ImportError:
        return False
    return tags.get(IS_PROMPT_TAG_KEY, "false").lower() == "true"


@dataclass
class RegisteredModelParams(BaseParams):
    is_prompt: bool

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "RegisteredModelParams":
        tags = arguments.get("tags") or {}
        is_prompt = _is_prompt(tags)
        return RegisteredModelParams(is_prompt=is_prompt)


@dataclass
class CreateRunParams(BaseParams):
    # Capture the set of currently imported packages at run creation time to
    # understand how MLflow is used together with other libraries. Collecting
    # this data at run creation ensures accuracy and completeness.
    imports: list[str]

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "CreateRunParams":
        imports = [pkg for pkg in PACKAGES_TO_CHECK_IMPORT if pkg in sys.modules]
        return CreateRunParams(imports=imports)


@dataclass
class CreateModelVersionParams(BaseParams):
    is_prompt: bool

    @classmethod
    def parse(cls, arguments: dict[str, Any]) -> "CreateModelVersionParams":
        tags = arguments.get("tags") or {}
        is_prompt = _is_prompt(tags)
        return CreateModelVersionParams(is_prompt=is_prompt)


PARAMS_MAPPING: dict[EventName, BaseParams] = {
    EventName.CREATE_LOGGED_MODEL: LoggedModelParams,
    EventName.CREATE_REGISTERED_MODEL: RegisteredModelParams,
    EventName.CREATE_RUN: CreateRunParams,
    EventName.CREATE_MODEL_VERSION: CreateModelVersionParams,
}


@dataclass
class Record:
    event_name: str
    timestamp_ns: int
    params: Optional[BaseParams] = None
    status: Status = Status.UNKNOWN
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "event_name": self.event_name,
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
    tracking_uri_scheme: Optional[str] = None


@dataclass
class TelemetryConfig:
    ingestion_url: str
    disable_events: set[str]
