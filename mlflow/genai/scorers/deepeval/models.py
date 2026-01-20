from __future__ import annotations

import json

from deepeval.models import LiteLLMModel
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import ValidationError

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    _invoke_databricks_serving_endpoint,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL


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
    """
    DeepEval model adapter for MLflow AI Gateway endpoints.

    Uses the MLflow AI Gateway to proxy requests to the configured backend.
    Routes requests through the tracking server's gateway endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        self._model_uri = f"gateway:/{endpoint_name}"
        super().__init__(model_name=self._model_uri)

    def load_model(self, **kwargs):
        return self

    def _get_gateway_base_url(self) -> str:
        """Get the gateway base URL from tracking URI."""
        from mlflow.environment_variables import MLFLOW_GATEWAY_URI
        from mlflow.tracking import get_tracking_uri
        from mlflow.utils.uri import append_to_uri_path, is_http_uri

        tracking_uri = MLFLOW_GATEWAY_URI.get() or get_tracking_uri()

        if not is_http_uri(tracking_uri):
            raise MlflowException(
                f"Gateway provider requires an HTTP(S) tracking URI, but got: '{tracking_uri}'. "
                "Please set MLFLOW_TRACKING_URI to a valid HTTP(S) URL "
                "(e.g., 'http://localhost:5000')."
            )

        return append_to_uri_path(tracking_uri, "gateway/mlflow/v1/")

    def generate(self, prompt: str, schema=None) -> str:
        import litellm

        api_base = self._get_gateway_base_url()
        # Use openai/ prefix for LiteLLM to use OpenAI-compatible format
        model = f"openai/{self._endpoint_name}"

        if schema is not None:
            json_prompt = _build_json_prompt_with_schema(prompt, schema)
        else:
            json_prompt = prompt

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": json_prompt}],
            api_base=api_base,
            api_key="mlflow-gateway-auth",  # Dummy key, gateway handles auth
        )

        output = response.choices[0].message.content

        if schema is not None:
            return _parse_json_output_with_schema(output.strip(), schema)
        return output

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self._model_uri


class DatabricksServingEndpointDeepEvalLLM(DeepEvalBaseLLM):
    """
    DeepEval model adapter for Databricks serving endpoints.

    Uses the model serving API via _invoke_databricks_serving_endpoint.
    """

    def __init__(self, endpoint_name: str):
        self._endpoint_name = endpoint_name
        super().__init__(model_name=f"databricks:/{endpoint_name}")

    def load_model(self, **kwargs):
        return self

    def generate(self, prompt: str, schema=None) -> str:
        if schema is not None:
            # TODO: Use response_format parameter once Databricks serving endpoints support it
            json_prompt = _build_json_prompt_with_schema(prompt, schema)
            output = _invoke_databricks_serving_endpoint(
                model_name=self._endpoint_name,
                prompt=json_prompt,
                num_retries=3,
                response_format=None,
            )
            return _parse_json_output_with_schema(output.response, schema)
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


def create_deepeval_model(model_uri: str):
    if model_uri == "databricks":
        return DatabricksDeepEvalLLM()
    elif model_uri.startswith("databricks:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return DatabricksServingEndpointDeepEvalLLM(endpoint_name)
    elif model_uri.startswith("gateway:/"):
        endpoint_name = model_uri.split(":", 1)[1].removeprefix("/")
        return GatewayDeepEvalLLM(endpoint_name)
    elif ":" in model_uri:
        # LiteLLM model format with provider: provider:/model_name (e.g., openai:/gpt-4)
        provider, model_name = model_uri.split(":", 1)
        model_name = model_name.removeprefix("/")
        return LiteLLMModel(model=f"{provider}/{model_name}")
    else:
        raise MlflowException.invalid_parameter_value(
            f"Invalid model_uri format: '{model_uri}'. "
            f"Must be 'databricks' or include a provider prefix (e.g., 'openai:/gpt-4')."
        )
