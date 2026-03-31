from __future__ import annotations

import json

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import ValidationError

from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.scorers.llm_backend import MLflowLLMBackend


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


class GatewayDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval model adapter using the shared MLflow LLM backend."""

    def __init__(self, backend: MLflowLLMBackend):
        super().__init__(model_name=backend.model_name)
        self._backend = backend

    def load_model(self, **kwargs):
        return self

    # Return type is str when schema is None, or a validated schema instance when
    # schema is provided. The -> str annotation matches DeepEvalBaseLLM's abstract
    # method signature; DeepEval's own LiteLLMModel follows the same convention.
    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            prompt = _build_json_prompt_with_schema(prompt, schema)

        response = self._backend.complete(prompt)

        if schema is not None:
            return _parse_json_output_with_schema(response.strip(), schema)
        return response

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self._backend.model_name


def create_deepeval_model(model_uri: str):
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        # Databricks managed judge uses the backend's databricks route
        return GatewayDeepEvalLLM(MLflowLLMBackend(model_uri))

    return GatewayDeepEvalLLM(MLflowLLMBackend(model_uri))
