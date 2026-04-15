from __future__ import annotations

import json
import typing as t

from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import InstructorBaseRagasLLM

from mlflow.genai.judges.utils.parsing_utils import _strip_markdown_code_blocks
from mlflow.genai.scorers.llm_backend import ScorerLLMClient

T = t.TypeVar("T", bound=BaseModel)


class MlflowRagasLLM(InstructorBaseRagasLLM):
    """RAGAS LLM adapter backed by the shared scorer LLM client.

    Routes through native providers when available, falls back to litellm.
    Handles structured output via JSON prompt injection and response parsing.
    """

    def __init__(self, backend: ScorerLLMClient):
        super().__init__()
        self.is_async = False
        self._backend = backend

    def generate(self, prompt: str, response_model: type[T]) -> T:
        full_prompt = _build_json_prompt(prompt, response_model)
        response = self._backend.complete_prompt(full_prompt)
        return _parse_json_response(response, response_model)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return self.generate(prompt, response_model)

    def get_model_name(self) -> str:
        return self._backend.model_name


def create_ragas_model(model_uri: str):
    backend = ScorerLLMClient(model_uri)

    if backend.is_native:
        return MlflowRagasLLM(backend)

    import instructor
    import litellm
    from ragas.llms.litellm_llm import LiteLLMStructuredLLM

    client = instructor.from_litellm(litellm.acompletion)
    return LiteLLMStructuredLLM(
        client=client,
        model=backend.model_name,
        provider=backend.provider,
        drop_params=True,
    )


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
