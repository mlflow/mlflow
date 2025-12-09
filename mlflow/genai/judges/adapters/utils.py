"""Utility functions for judge adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.genai.judges.adapters.base_adapter import BaseJudgeAdapter
    from mlflow.types.llm import ChatMessage

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    DatabricksManagedJudgeAdapter,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    DatabricksServingEndpointAdapter,
)
from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
from mlflow.genai.judges.adapters.litellm_adapter import LiteLLMAdapter
from mlflow.protos.databricks_pb2 import BAD_REQUEST


def get_adapter(
    model_uri: str,
    prompt: str | list["ChatMessage"],
) -> "BaseJudgeAdapter":
    """
    Factory function to get the appropriate adapter for a given model configuration. Tries adapters
    in order of priority.

    Args:
        model_uri: The full model URI (e.g., "openai:/gpt-4", "databricks").
        prompt: The prompt to evaluate (string or list of ChatMessages).

    Returns:
        An instance of the appropriate adapter.

    Raises:
        MlflowException: If no suitable adapter is found.
    """
    adapters = [
        DatabricksManagedJudgeAdapter,
        DatabricksServingEndpointAdapter,
        LiteLLMAdapter,
        GatewayAdapter,
    ]

    for adapter_class in adapters:
        if adapter_class.is_applicable(model_uri=model_uri, prompt=prompt):
            return adapter_class()

    raise MlflowException(
        f"No suitable adapter found for model_uri='{model_uri}'. "
        "Some providers may require LiteLLM to be invoked. "
        "Please install it with: `pip install litellm`",
        error_code=BAD_REQUEST,
    )
