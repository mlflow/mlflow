import sys
from abc import ABC, abstractmethod
from typing import Any, Optional

from mlflow.telemetry.constant import PACKAGES_TO_CHECK_IMPORT
from mlflow.telemetry.schemas import (
    BaseParams,
    CreateRunParams,
    LoggedModelParams,
    RegisteredModelParams,
)


class TelemetryParser(ABC):
    @classmethod
    @abstractmethod
    def extract_params(cls, arguments: dict[str, Any]) -> Optional[BaseParams]:
        """
        Extract the parameters from the function call.

        Args:
            arguments: The arguments passed to the function.

        Returns:
            The parsed params that extend BaseParams.
        """


class LoggedModelParser(TelemetryParser):
    @classmethod
    def extract_params(cls, arguments: dict[str, Any]) -> Optional[LoggedModelParams]:
        if flavor := arguments.get("flavor"):
            return LoggedModelParams(flavor=flavor.removeprefix("mlflow."))
        return None


class RegisteredModelParser(TelemetryParser):
    @classmethod
    def extract_params(cls, arguments: dict[str, Any]) -> Optional[RegisteredModelParams]:
        tags = arguments.get("tags") or {}
        is_prompt = False
        try:
            from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
        except ImportError:
            pass
        else:
            is_prompt = tags.get(IS_PROMPT_TAG_KEY, "false").lower() == "true"
        return RegisteredModelParams(is_prompt=is_prompt)


class CreateRunParser(TelemetryParser):
    @classmethod
    def extract_params(cls, arguments: dict[str, Any]) -> Optional[CreateRunParams]:
        imports = [pkg for pkg in PACKAGES_TO_CHECK_IMPORT if pkg in sys.modules]
        return CreateRunParams(imports=imports)


API_PARSER_MAPPING: dict[str, TelemetryParser] = {
    "create_logged_model": LoggedModelParser,
    "create_registered_model": RegisteredModelParser,
    "create_run": CreateRunParser,
}
