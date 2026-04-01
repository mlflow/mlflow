from __future__ import annotations

import json

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import ValidationError

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.metrics.genai.model_utils import (
    _call_llm_provider_api,
    _get_provider_instance,
    _parse_model_uri,
)


def _build_json_prompt_with_schema(prompt: str, schema) -> str:
    return (
        f"{prompt}\n\n"
        f"IMPORTANT: Return your response as valid JSON matching this schema: "
        f"{schema.model_json_schema()}\n"
        f"Return ONLY the JSON object, no additional text or markdown formatting."
    )


def _parse_json_output_with_schema(output: str, schema):
    try:
        json_data = json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON output: {e}\nOutput: {output}")

    try:
        return schema(**json_data)
    except ValidationError as e:
        raise ValueError(f"Failed to validate output against schema: {e}\nOutput: {output}")
    except TypeError as e:
        raise ValueError(f"Failed to instantiate schema with data: {e}\nOutput: {output}")


class DatabricksDeepEvalLLM(DeepEvalBaseLLM):
    """
    DeepEval model adapter for Databricks managed judge.

    Uses the default Databricks endpoint via call_chat_completions.
    """

    def __init__(self):
        super().__init__(model_name=_DATABRICKS_DEFAULT_JUDGE_MODEL)

    def load_model(self, **kwargs):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            # TODO: Add support for structured outputs once the Databricks endpoint supports it
            json_prompt = _build_json_prompt_with_schema(prompt, schema)
            result = call_chat_completions(user_prompt=json_prompt, system_prompt="")
            return _parse_json_output_with_schema(result.output.strip(), schema)
        else:
            result = call_chat_completions(user_prompt=prompt, system_prompt="")
            return result.output

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return _DATABRICKS_DEFAULT_JUDGE_MODEL


class GatewayDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval model adapter using MLflow Gateway providers.

    Uses the native provider infrastructure (_call_llm_provider_api) instead
    of litellm. Handles structured output via JSON prompt injection and
    response parsing (same approach as DatabricksDeepEvalLLM).
    """

    def __init__(self, provider: str, model_name: str):
        super().__init__(model_name=f"{provider}/{model_name}")
        self._provider = provider
        self._model_name = model_name

    def load_model(self, **kwargs):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        # Return type is str when schema is None, or a validated schema instance when
        # schema is provided. The -> str annotation matches DeepEvalBaseLLM's abstract
        # method signature; DeepEval's own LiteLLMModel follows the same convention.
        if schema is not None:
            prompt = _build_json_prompt_with_schema(prompt, schema)

        response = _call_llm_provider_api(
            self._provider,
            self._model_name,
            input_data=prompt,
        )

        if schema is not None:
            return _parse_json_output_with_schema(response.strip(), schema)
        return response

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return f"{self._provider}/{self._model_name}"


def create_deepeval_model(model_uri: str):
    if model_uri == "databricks":
        return DatabricksDeepEvalLLM()

    provider, model_name = _parse_model_uri(model_uri)

    # Use native gateway provider if _get_provider_instance can construct it,
    # otherwise fall back to litellm
    try:
        _get_provider_instance(provider, model_name)
    except MlflowException:
        pass
    else:
        return GatewayDeepEvalLLM(provider, model_name)

    from deepeval.models import LiteLLMModel

    return LiteLLMModel(
        model=f"{provider}/{model_name}",
        generation_kwargs={"drop_params": True},
    )
