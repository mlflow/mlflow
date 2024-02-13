from abc import abstractmethod
import pandas as pd
import requests
from typing import Any, Dict, List, Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc import PythonModel

LLM_V1_PREFIX = "llm/v1/"
LLM_ENDPOINT_TYPE_CHAT = "chat"
LLM_ENDPOINT_TYPE_COMPLETION = "completion"
SUPPORTED_ENDPOINT_TYPES = [LLM_ENDPOINT_TYPE_CHAT, LLM_ENDPOINT_TYPE_COMPLETION]


HINT_MSG = (
    "Invalid response format:\n{response}\n MLflow Evaluation expects OpenAI-compatible response "
    "format for chat and completion models. Please refer to OpenAI documentation: "
    "https://platform.openai.com/docs/api-reference/introduction for the expected format.\n\n"
    "HINT: If your endpoint has a different response format, you can still use  it for evaluation by "
    "defining a custom prediction function. For more details, please refer to the MLflow documentation: "
    "https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#evaluating-with-a-custom-function "
)

class _ServedLLMEndpointModel(PythonModel):
    """
    """
    def __init__(self,
                 endpoint: str,
                 default_params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None):
        self.endpoint = endpoint
        self.default_params = default_params or {}
        self.headers = headers or {}

    @abstractmethod
    def convert_input(self, input_data: str, params) -> Any:
        pass

    @abstractmethod
    def extract_output(self, response) -> str:
        pass

    def predict(self, context, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> List[str]:
        headers = {"Content-Type": "application/json", **(self.headers or {})}
        params = {**self.default_params, **(params or {})}

        if not "inputs" in model_input.columns:
            raise MlflowException(f"Invalid input column: {model_input.columns}. The input column for "
                                "evaluating a chat or completion model endpoint must be named 'inputs'")

        predictions = []
        for input_data in model_input["inputs"]:
            if not isinstance(input_data, str):
                raise MlflowException(f"Invalid input column type: {type(input_data), input_data}. The input column for "
                                    "evaluating a served LLM model must contain string values only")

            request = self.convert_input(input_data, params)

            response = requests.post(self.endpoint, json=request, headers=headers)
            response.raise_for_status()

            text = self.extract_output(response.json())
            predictions.append(text)

        return predictions


class _ServedChatEndpointModel(_ServedLLMEndpointModel):
    def convert_input(self, input_data: str, params) -> Dict[str, Any]:
        return {
            "messages": [{
                "content": input_data,
                "role": "user",
                "name": "User",
            }],
            **params
        }

    def extract_output(self, response) -> str:
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, TypeError) as e:
            raise MlflowException(HINT_MSG.format(response=response)) from e
        return text


class _ServedCompletionEndpointModel(_ServedLLMEndpointModel):
    def convert_input(self, input_data: str, params) -> Dict[str, Any]:
        return {
            "prompt": input_data,
            **params
        }

    def extract_output(self, response) -> str:
        try:
            text = response["choices"][0]["text"]
        except TypeError as e:
            raise MlflowException(HINT_MSG.format(response=response),
                                  error_code=INTERNAL_ERROR) from e
        return text


def get_model_from_llm_endpoint_url(
    endpoint: str,
    endpoint_type: str,
    params: Optional[Dict[str, Any]]=None,
    headers: Optional[Dict[str, str]]=None):
    """

    Args:

    Returns:
    """
    from mlflow.pyfunc.model import _PythonModelPyfuncWrapper

    # Also accepts endpoint type with "llm/v1/" prefix e.g. "llm/v1/chat"
    if endpoint_type.startswith(LLM_V1_PREFIX):
        endpoint_type = endpoint_type[len(LLM_V1_PREFIX):]

    if endpoint_type == LLM_ENDPOINT_TYPE_CHAT:
        python_model = _ServedChatEndpointModel(endpoint, params, headers)
    elif endpoint_type == LLM_ENDPOINT_TYPE_COMPLETION:
        python_model = _ServedCompletionEndpointModel(endpoint, params, headers)
    else:
        raise MlflowException(f"Invalid endpoint type: {endpoint_type}. Please specify one of "
                                f"{SUPPORTED_ENDPOINT_TYPES + [LLM_V1_PREFIX + t for t in SUPPORTED_ENDPOINT_TYPES]}")

    return _PythonModelPyfuncWrapper(python_model, None, None)
