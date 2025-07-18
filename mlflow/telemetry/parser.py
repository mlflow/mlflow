from abc import ABC, abstractmethod
from typing import Any, Optional

from mlflow.telemetry.schemas import (
    BaseParams,
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
        flavor = arguments.get("flavor")
        flavor = flavor.removeprefix("mlflow.") if flavor else "custom"
        return LoggedModelParams(flavor=flavor)


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


API_PARSER_MAPPING: dict[str, TelemetryParser] = {
    "create_logged_model": LoggedModelParser,
    "create_registered_model": RegisteredModelParser,
}
