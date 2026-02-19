from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.6.0")
class BaseAgentValidator:
    """Base validator class with common validation methods"""

    def validate_pydantic(self, pydantic_class: type[BaseModel], data: Any) -> None:
        """Generic pydantic validator that throws an error if the data is invalid"""
        if isinstance(data, pydantic_class):
            return
        try:
            if isinstance(data, BaseModel):
                pydantic_class(**data.model_dump())
                return
            pydantic_class(**data)
        except Exception as e:
            raise ValueError(f"Invalid data for {pydantic_class.__name__}: {e}")

    def validate_dataclass(self, dataclass_class: Any, data: Any) -> None:
        """Generic dataclass validator that throws an error if the data is invalid"""
        if isinstance(data, dataclass_class):
            return
        try:
            dataclass_class(**data)
        except Exception as e:
            raise ValueError(f"Invalid data for {dataclass_class.__name__}: {e}")

    def validate_and_convert_request(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def validate_and_convert_result(self, result: Any, stream: bool = False) -> dict[str, Any]:
        # Base implementation doesn't use stream parameter, but subclasses do
        if isinstance(result, BaseModel):
            return result.model_dump(exclude_none=True)
        elif is_dataclass(result):
            return asdict(result)
        elif isinstance(result, dict):
            return result
        else:
            raise ValueError(
                f"Result needs to be a pydantic model, dataclass, or dict. "
                f"Unsupported result type: {type(result)}, result: {result}"
            )


@experimental(version="3.6.0")
class ResponsesAgentValidator(BaseAgentValidator):
    def validate_and_convert_request(self, data: dict[str, Any]) -> ResponsesAgentRequest:
        self.validate_pydantic(ResponsesAgentRequest, data)
        return ResponsesAgentRequest(**data)

    def validate_and_convert_result(self, result: Any, stream: bool = False) -> dict[str, Any]:
        if stream:
            self.validate_pydantic(ResponsesAgentStreamEvent, result)
        else:
            self.validate_pydantic(ResponsesAgentResponse, result)

        return super().validate_and_convert_result(result, stream)
