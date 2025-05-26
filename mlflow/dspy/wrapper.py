import importlib.metadata
import json
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from packaging.version import Version

if TYPE_CHECKING:
    import dspy

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
)
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import DataType, Schema

_INVALID_SIZE_MESSAGE = (
    "Dspy model doesn't support batch inference or empty input. Please provide a single input."
)


class DspyModelWrapper(PythonModel):
    """MLflow PyFunc wrapper class for Dspy models.

    This wrapper serves two purposes:
        - It stores the Dspy model along with dspy global settings, which are required for seamless
            saving and loading.
        - It provides a `predict` method so that it can be loaded as an MLflow pyfunc, which is
            used at serving time.
    """

    def __init__(
        self,
        model: "dspy.Module",
        dspy_settings: dict[str, Any],
        model_config: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.dspy_settings = dspy_settings
        self.model_config = model_config or {}
        self.output_schema: Optional[Schema] = None

    def predict(self, inputs: Any, params: Optional[dict[str, Any]] = None):
        import dspy

        converted_inputs = self._get_model_input(inputs)

        with dspy.context(**self.dspy_settings):
            if isinstance(converted_inputs, dict):
                # We pass a dict as keyword args and don't allow DSPy models
                # to receive a single dict.
                return self.model(**converted_inputs).toDict()
            else:
                return self.model(converted_inputs).toDict()

    def predict_stream(self, inputs: Any, params=None):
        import dspy

        converted_inputs = self._get_model_input(inputs)

        self._validate_streaming()

        stream_listeners = [
            dspy.streaming.StreamListener(signature_field_name=spec.name)
            for spec in self.output_schema
        ]
        stream_model = dspy.streamify(
            self.model,
            stream_listeners=stream_listeners,
            async_streaming=False,
            include_final_prediction_in_output_stream=False,
        )

        if isinstance(converted_inputs, dict):
            outputs = stream_model(**converted_inputs)
        else:
            outputs = stream_model(converted_inputs)

        with dspy.context(**self.dspy_settings):
            for output in outputs:
                if is_dataclass(output):
                    yield asdict(output)
                elif isinstance(output, dspy.Prediction):
                    yield output.toDict()
                else:
                    yield output

    def _get_model_input(self, inputs: Any) -> Union[str, dict[str, Any]]:
        """Convert the PythonModel input into the DSPy program input

        Examples of expected conversions:
        - str -> str
        - dict -> dict
        - np.ndarray with one element -> single element
        - pd.DataFrame with one row and string column -> single row dict
        - pd.DataFrame with one row and non-string column -> single element
        - list -> raises an exception
        - np.ndarray with more than one element -> raises an exception
        - pd.DataFrame with more than one row -> raises an exception
        """
        import numpy as np
        import pandas as pd

        supported_input_types = (np.ndarray, pd.DataFrame, str, dict)
        if not isinstance(inputs, supported_input_types):
            raise MlflowException(
                f"`inputs` must be one of: {[x.__name__ for x in supported_input_types]}, but "
                f"received type: {type(inputs)}.",
                INVALID_PARAMETER_VALUE,
            )
        if isinstance(inputs, pd.DataFrame):
            if len(inputs) != 1:
                raise MlflowException(
                    _INVALID_SIZE_MESSAGE,
                    INVALID_PARAMETER_VALUE,
                )
            if all(isinstance(col, str) for col in inputs.columns):
                inputs = inputs.to_dict(orient="records")[0]
            else:
                inputs = inputs.values[0]
        if isinstance(inputs, np.ndarray):
            if len(inputs) != 1:
                raise MlflowException(
                    _INVALID_SIZE_MESSAGE,
                    INVALID_PARAMETER_VALUE,
                )
            inputs = inputs[0]

        return inputs

    def _validate_streaming(
        self,
    ):
        if Version(importlib.metadata.version("dspy")) <= Version("2.6.23"):
            raise MlflowException(
                "Streaming API is only supported in dspy 2.6.24 or later. "
                "Please upgrade your dspy version."
            )

        if self.output_schema is None:
            raise MlflowException(
                "Output schema of the DSPy model is not set. Please log your DSPy "
                "model with `signature` or `input_example` to use streaming API."
            )

        if any(spec.type != DataType.string for spec in self.output_schema):
            raise MlflowException(
                f"All output fields must be string to use streaming API. Got {self.output_schema}."
            )


class DspyChatModelWrapper(DspyModelWrapper):
    """MLflow PyFunc wrapper class for Dspy chat models."""

    def predict(self, inputs: Any, params: Optional[dict[str, Any]] = None):
        import dspy

        converted_inputs = self._get_model_input(inputs)

        # `dspy.settings` cannot be shared across threads, so we are setting the context at every
        # predict call.
        with dspy.context(**self.dspy_settings):
            outputs = self.model(converted_inputs)

        choices = []
        if isinstance(outputs, str):
            choices.append(self._construct_chat_message("assistant", outputs))
        elif isinstance(outputs, dict):
            role = outputs.get("role", "assistant")
            choices.append(self._construct_chat_message(role, json.dumps(outputs)))
        elif isinstance(outputs, dspy.Prediction):
            choices.append(self._construct_chat_message("assistant", json.dumps(outputs.toDict())))
        elif isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict):
                    role = output.get("role", "assistant")
                    choices.append(self._construct_chat_message(role, json.dumps(outputs)))
                elif isinstance(output, dspy.Prediction):
                    role = output.get("role", "assistant")
                    choices.append(self._construct_chat_message(role, json.dumps(outputs.toDict())))
                else:
                    raise MlflowException(
                        f"Unsupported output type: {type(output)}. To log a DSPy model with task "
                        "'llm/v1/chat', the DSPy model must return a dict, a dspy.Prediction, or a "
                        "list of dicts or dspy.Prediction.",
                        INVALID_PARAMETER_VALUE,
                    )
        else:
            raise MlflowException(
                f"Unsupported output type: {type(outputs)}. To log a DSPy model with task "
                "'llm/v1/chat', the DSPy model must return a dict, a dspy.Prediction, or a list of "
                "dicts or dspy.Prediction.",
                INVALID_PARAMETER_VALUE,
            )

        return {"choices": choices}

    def predict_stream(self, inputs: Any, params=None):
        raise NotImplementedError(
            "Streaming is not supported for DSPy model with task 'llm/v1/chat'."
        )

    def _get_model_input(self, inputs: Any) -> Union[str, list[dict[str, Any]]]:
        import pandas as pd

        if isinstance(inputs, dict):
            return inputs["messages"]
        if isinstance(inputs, pd.DataFrame):
            return inputs.messages[0]

        raise MlflowException(
            f"Unsupported input type: {type(inputs)}. To log a DSPy model with task "
            "'llm/v1/chat', the input must be a dict or a pandas DataFrame.",
            INVALID_PARAMETER_VALUE,
        )

    def _construct_chat_message(self, role: str, content: str) -> dict[str, Any]:
        return {
            "index": 0,
            "message": {
                "role": role,
                "content": content,
            },
            "finish_reason": "stop",
        }
