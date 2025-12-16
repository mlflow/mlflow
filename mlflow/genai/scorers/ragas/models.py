from __future__ import annotations

import instructor
import litellm
from langchain_core.outputs import Generation, LLMResult
from ragas.llms import BaseRagasLLM
from ragas.llms.litellm_llm import LiteLLMStructuredLLM

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL


class DatabricksRagasLLM(BaseRagasLLM):
    """
    RAGAS LLM adapter for Databricks managed judge.

    Uses the default Databricks endpoint via call_chat_completions.
    """

    def __init__(self):
        super().__init__()

    def generate_text(self, prompt: str, **kwargs) -> str:
        # Convert LangChain StringPromptValue to string if needed
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        result = call_chat_completions(user_prompt=prompt, system_prompt="")
        return result.output

    async def agenerate_text(self, prompt: str, **kwargs):
        text = self.generate_text(prompt, **kwargs)
        generation = Generation(text=text)
        return LLMResult(generations=[[generation]])

    def get_model_name(self) -> str:
        return _DATABRICKS_DEFAULT_JUDGE_MODEL

    def is_finished(self, result=None) -> bool:
        return True


class DatabricksServingEndpointRagasLLM(BaseRagasLLM):
    """
    RAGAS LLM adapter for Databricks serving endpoints.

    Uses the model serving API via _invoke_databricks_serving_endpoint.
    """

    def __init__(self, endpoint_name: str):
        super().__init__()
        self._endpoint_name = endpoint_name

    def generate_text(self, prompt: str, **kwargs) -> str:
        # Convert LangChain StringPromptValue to string if needed
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        output = _invoke_databricks_serving_endpoint(
            model_name=self._endpoint_name,
            prompt=prompt,
            num_retries=3,
            response_format=None,
        )
        return output.response

    async def agenerate_text(self, prompt: str, **kwargs):
        text = self.generate_text(prompt, **kwargs)
        generation = Generation(text=text)
        return LLMResult(generations=[[generation]])

    def get_model_name(self) -> str:
        return f"databricks:/{self._endpoint_name}"

    def is_finished(self, result=None) -> bool:
        return True


def create_ragas_model(model_uri: str):
    """
    Create a RAGAS LLM adapter from a model URI.

    Args:
        model_uri: Model URI in one of these formats:
            - "databricks" - Use default Databricks managed judge
            - "databricks:/endpoint" - Use Databricks serving endpoint
            - "provider:/model" - Use LiteLLM (e.g., "openai:/gpt-4")

    Returns:
        A RAGAS-compatible LLM adapter

    Raises:
        MlflowException: If the model URI format is invalid
    """
    if model_uri == "databricks":
        return DatabricksRagasLLM()
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return DatabricksServingEndpointRagasLLM(endpoint_name)
    elif ":" in model_uri:
        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")
        client = instructor.from_litellm(litellm.completion)
        return LiteLLMStructuredLLM(
            client=client,
            model=f"{provider}/{model_name}",
            provider=provider,
        )
    else:
        raise MlflowException.invalid_parameter_value(
            f"Invalid model_uri format: '{model_uri}'. "
            f"Must be 'databricks' or include a provider prefix (e.g., 'openai:/gpt-4') "
            f"or a Databricks serving endpoint (e.g., 'databricks:/<endpoint_name>')."
        )
