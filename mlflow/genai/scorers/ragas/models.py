from __future__ import annotations

import json
import typing as t

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import InstructorBaseRagasLLM

from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.utils.parsing_utils import _strip_markdown_code_blocks
from mlflow.genai.scorers.llm_backend import MLflowLLMBackend

T = t.TypeVar("T", bound=BaseModel)


class GatewayRagasLLM(InstructorBaseRagasLLM):
    """RAGAS LLM adapter using the shared MLflow LLM backend."""

    def __init__(self, backend: MLflowLLMBackend):
        super().__init__()
        self.is_async = False
        self._backend = backend

    def generate(self, prompt: str, response_model: type[T]) -> T:
        full_prompt = _build_json_prompt(prompt, response_model)
        response = self._backend.complete(full_prompt)
        return _parse_json_response(response, response_model)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return self.generate(prompt, response_model)

    def get_model_name(self) -> str:
        return self._backend.model_name


def create_ragas_model(model_uri: str):
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        return GatewayRagasLLM(MLflowLLMBackend(model_uri))

    return GatewayRagasLLM(MLflowLLMBackend(model_uri))


def create_default_embeddings():
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
