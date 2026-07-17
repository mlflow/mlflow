from __future__ import annotations

import json
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import ValidationError

from mlflow.genai.scorers.llm_backend import ScorerLLMClient


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


class MlflowDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval model adapter backed by the shared scorer LLM client.

    Routes through native providers when available, falls back to litellm.
    Handles structured output via JSON prompt injection and response parsing.
    """

    def __init__(self, backend: ScorerLLMClient, model_kwargs: dict[str, Any] | None = None):
        super().__init__(model_name=backend.model_name)
        self._backend = backend
        self._model_kwargs = model_kwargs or {}

    def load_model(self, **kwargs):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            prompt = _build_json_prompt_with_schema(prompt, schema)

        response = self._backend.complete_prompt(prompt, **self._model_kwargs)

        if schema is not None:
            return _parse_json_output_with_schema(response.strip(), schema)
        return response

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self._backend.model_name


def create_deepeval_model(model_uri: str, model_kwargs: dict[str, Any] | None = None):
    backend = ScorerLLMClient(model_uri)

    if backend.is_native:
        return MlflowDeepEvalLLM(backend, model_kwargs=model_kwargs)

    from deepeval.models import LiteLLMModel

    # DeepEval's LiteLLMModel.__init__ strips `temperature` from `generation_kwargs`
    # and reads it only from the top-level `temperature` constructor arg. If a user
    # passes temperature in model_kwargs we have to lift it out here, otherwise the
    # value is silently dropped and the model falls back to its default (0.0).
    extra = dict(model_kwargs) if model_kwargs else {}
    temperature = extra.pop("temperature", None)
    generation_kwargs = {"drop_params": True, **extra}

    kwargs: dict[str, Any] = {
        "model": backend.model_name,
        "generation_kwargs": generation_kwargs,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    return LiteLLMModel(**kwargs)
