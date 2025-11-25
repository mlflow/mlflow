from __future__ import annotations

import json

from deepeval.models.base_model import DeepEvalBaseLLM

from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL


class DatabricksDeepEvalLLM(DeepEvalBaseLLM):
    """
    DeepEval model adapter for Databricks managed judge.

    Uses the default Databricks endpoint via call_chat_completions.
    """

    def __init__(self):
        super().__init__(model=_DATABRICKS_DEFAULT_JUDGE_MODEL)

    def load_model(self, **kwargs):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            # TODO: Add support for structured outputs once the Databricks endpoint supports it
            json_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: Return your response as valid JSON matching this schema: "
                f"{schema.model_json_schema()}\n"
                f"Return ONLY the JSON object, no additional text or markdown formatting."
            )
            result = call_chat_completions(user_prompt=json_prompt, system_prompt="")
            output = result.output.strip()

            try:
                json_data = json.loads(output)
                return schema(**json_data)
            except (json.JSONDecodeError, Exception) as e:
                raise ValueError(f"Failed to parse structured output: {e}\nOutput: {output}")
        else:
            result = call_chat_completions(user_prompt=prompt, system_prompt="")
            return result.output

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return _DATABRICKS_DEFAULT_JUDGE_MODEL

    def should_use_azure_openai(self) -> bool:
        return False


class DatabricksServingEndpointDeepEvalLLM(DeepEvalBaseLLM):
    """
    DeepEval model adapter for Databricks serving endpoints.

    Uses the model serving API via _invoke_databricks_serving_endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        super().__init__(model=f"databricks:/{endpoint_name}")

    def load_model(self, **kwargs):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            # TODO: Use response_format parameter once Databricks serving endpoints support it
            json_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: Return your response as valid JSON matching this schema: "
                f"{schema.model_json_schema()}\n"
                f"Return ONLY the JSON object, no additional text or markdown formatting."
            )
            output = _invoke_databricks_serving_endpoint(
                model_name=self._endpoint_name,
                prompt=json_prompt,
                num_retries=3,
                response_format=None,
            )

            try:
                json_data = json.loads(output.response)
                return schema(**json_data)
            except (json.JSONDecodeError, Exception) as e:
                raise ValueError(
                    f"Failed to parse structured output: {e}\nOutput: {output.response}"
                )
        else:
            output = _invoke_databricks_serving_endpoint(
                model_name=self._endpoint_name,
                prompt=prompt,
                num_retries=3,
                response_format=None,
            )
            return output.response

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return f"databricks:/{self._endpoint_name}"

    def should_use_azure_openai(self) -> bool:
        return False
