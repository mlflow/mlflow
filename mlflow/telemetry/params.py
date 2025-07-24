import json
import sys
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Optional

from mlflow.telemetry.constant import PACKAGES_TO_CHECK_IMPORT


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


PARAMS_MAPPING: dict[str, BaseParams] = {
    "create_logged_model": LoggedModelParams,
    "create_registered_model": RegisteredModelParams,
    "create_run": CreateRunParams,
    "create_model_version": CreateModelVersionParams,
}
