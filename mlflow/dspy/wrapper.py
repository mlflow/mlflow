import json
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models.rag_signatures import (
    ChainCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
)
from mlflow.pyfunc import PythonModel


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
        model: "dspy.Module",  # noqa: F821
        dspy_settings: "dsp.utils.settings.Settings",  # noqa: F821
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.dspy_settings = dspy_settings
        self.model_config = model_config or {}

    def predict(self, inputs: Any, params: Optional[Dict[str, Any]] = None):
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
            # Return the output as a dict for serving simplicity.
            return self.model(inputs).toDict()

        if isinstance(inputs, dict):
            return self.model(**inputs).toDict()
        if isinstance(inputs, str):
            return self.model(inputs).toDict()


class DspyChatModelWrapper(DspyModelWrapper):
    def predict(self, inputs: ChatCompletionRequest, params: Optional[Dict[str, Any]] = None):
        import dspy

        converted_inputs = [asdict(data) for data in inputs.messages]
        outputs = self.model(converted_inputs)

        choices = []
        if isinstance(outputs, dict):
            role = outputs.get("role", "assistant")
            choices.append(
                ChainCompletionChoice(message=Message(role=role, content=json.dumps(outputs)))
            )
        elif isinstance(outputs, dspy.Prediction):
            choices.append(
                ChainCompletionChoice(message=Message(content=json.dumps(outputs.toDict())))
            )
        elif isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, dict):
                    role = output.get("role", "assistant")
                    choices.append(
                        ChainCompletionChoice(
                            message=Message(role=role, content=json.dumps(output))
                        )
                    )
                elif isinstance(output, dspy.Prediction):
                    choices.append(
                        ChainCompletionChoice(message=Message(content=json.dumps(output.toDict())))
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

        return ChatCompletionResponse(choices=choices)
