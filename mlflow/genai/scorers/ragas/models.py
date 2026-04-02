from __future__ import annotations

import json
import typing as t

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import InstructorBaseRagasLLM

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.utils.parsing_utils import _strip_markdown_code_blocks
from mlflow.metrics.genai.model_utils import (
    _call_llm_provider_api,
    _get_provider_instance,
    _parse_model_uri,
)

T = t.TypeVar("T", bound=BaseModel)


class DatabricksRagasLLM(InstructorBaseRagasLLM):
    """
    RAGAS LLM adapter for Databricks managed judge.

    Uses the default Databricks endpoint via call_chat_completions.
    """

    def __init__(self):
        super().__init__()
        self.is_async = False

    def generate(self, prompt: str, response_model: type[T]) -> T:
        full_prompt = _build_json_prompt(prompt, response_model)
        result = call_chat_completions(user_prompt=full_prompt, system_prompt="")
        return _parse_json_response(result.output, response_model)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return self.generate(prompt, response_model)

    def get_model_name(self) -> str:
        return _DATABRICKS_DEFAULT_JUDGE_MODEL


class GatewayRagasLLM(InstructorBaseRagasLLM):
    """RAGAS LLM adapter using MLflow Gateway providers.

    Uses the native provider infrastructure (_call_llm_provider_api) instead
    of litellm. Handles structured output via JSON prompt injection and
    response parsing (same approach as DatabricksRagasLLM).
    """

    def __init__(self, provider: str, model_name: str):
        super().__init__()
        self.is_async = False
        self._provider = provider
        self._model_name = model_name

    def generate(self, prompt: str, response_model: type[T]) -> T:
        full_prompt = _build_json_prompt(prompt, response_model)
        response = _call_llm_provider_api(
            self._provider,
            self._model_name,
            input_data=full_prompt,
        )
        return _parse_json_response(response, response_model)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return self.generate(prompt, response_model)

    def get_model_name(self) -> str:
        return f"{self._provider}/{self._model_name}"


def create_ragas_model(model_uri: str):
    """
    Create a RAGAS LLM adapter from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use Databricks serving endpoint
            - "gateway:/endpoint" - Use MLflow AI Gateway endpoint
            - "provider:/model" - Use native gateway provider or LiteLLM fallback

    Returns:
        A RAGAS-compatible LLM adapter

    Raises:
        MlflowException: If the model URI format is invalid
    """
    if model_uri == "databricks":
        return DatabricksRagasLLM()

    provider, model_name = _parse_model_uri(model_uri)

    # Use native gateway provider if _get_provider_instance can construct it,
    # otherwise fall back to litellm
    try:
        _get_provider_instance(provider, model_name)
    except MlflowException:
        pass
    else:
        return GatewayRagasLLM(provider, model_name)

    import litellm
    from ragas.llms.litellm_llm import LiteLLMStructuredLLM

    client = instructor.from_litellm(litellm.acompletion)
    return LiteLLMStructuredLLM(
        client=client,
        model=f"{provider}/{model_name}",
        provider=provider,
        drop_params=True,
    )


def create_default_embeddings():
    """
    Create default OpenAI embeddings for RAGAS metrics that require them.

    Returns:
        An OpenAIEmbeddings instance configured with a sync client.
    """
    return OpenAIEmbeddings(client=AsyncOpenAI())


def _build_json_prompt(prompt: str, response_model: type[T]) -> str:
    schema = response_model.model_json_schema()
    fields = schema.get("properties", {})
    field_desc = ", ".join(f'"{k}"' for k in fields.keys())
    return (
        f"{prompt}\n\n"
        f"OUTPUT FORMAT: Respond ONLY with a JSON object "
        f"containing these fields: {field_desc}, no other text. "
        f"Do not add markdown formatting to the response."
    )


def _parse_json_response(response: str, response_model: type[T]) -> T:
    text = _strip_markdown_code_blocks(response)
    try:
        return response_model.model_validate(json.loads(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON. Response was: {response}") from e
