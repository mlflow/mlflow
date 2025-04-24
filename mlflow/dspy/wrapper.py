import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import dspy

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
)
from mlflow.pyfunc import PythonModel

_logger = logging.getLogger(__name__)


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

    def predict(self, inputs: Any, params: Optional[dict[str, Any]] = None):
        import dspy
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
            inputs = inputs.values
        if isinstance(inputs, np.ndarray):
            flatten = inputs.reshape(-1)
            if len(flatten) > 1:
                raise MlflowException(
                    "Dspy model doesn't support multiple inputs or batch inference. Please "
                    "provide a single input.",
                    INVALID_PARAMETER_VALUE,
                )
            inputs = str(flatten[0])

        with dspy.context(**self.dspy_settings):
            if isinstance(inputs, dict):
                return self.model(**inputs).toDict()
            if isinstance(inputs, str):
                return self.model(inputs).toDict()


class DspyChatModelWrapper(DspyModelWrapper):
    """MLflow PyFunc wrapper class for Dspy chat models."""

    def predict(self, inputs: Any, params: Optional[dict[str, Any]] = None):
        import dspy
        import pandas as pd

        if isinstance(inputs, dict):
            converted_inputs = inputs["messages"]
        elif isinstance(inputs, pd.DataFrame):
            converted_inputs = inputs.messages[0]
        else:
            raise MlflowException(
                f"Unsupported input type: {type(inputs)}. To log a DSPy model with task "
                "'llm/v1/chat', the input must be a dict or a pandas DataFrame.",
                INVALID_PARAMETER_VALUE,
            )

        # `dspy.settings` cannot be shared across threads, so we are setting the context at every
        # predict call.
        with dspy.context(**self.dspy_settings):
            outputs = self.model(converted_inputs)

        choices = []
        if isinstance(outputs, str):
            choices.append(
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": outputs},
                    "finish_reason": "stop",
                }
            )
        elif isinstance(outputs, dict):
            role = outputs.get("role", "assistant")
            choices.append(
                {
                    "index": 0,
                    "message": {"role": role, "content": json.dumps(outputs)},
                    "finish_reason": "stop",
                }
            )
        elif isinstance(outputs, dspy.Prediction):
            choices.append(
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(outputs.toDict()),
                    },
                    "finish_reason": "stop",
                }
            )
        elif isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict):
                    role = output.get("role", "assistant")
                    choices.append(
                        {
                            "index": 0,
                            "message": {"role": role, "content": json.dumps(outputs)},
                            "finish_reason": "stop",
                        }
                    )
                elif isinstance(output, dspy.Prediction):
                    choices.append(
                        {
                            "index": 0,
                            "message": {
                                "role": role,
                                "content": json.dumps(outputs.toDict()),
                            },
                            "finish_reason": "stop",
                        }
                    )
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
