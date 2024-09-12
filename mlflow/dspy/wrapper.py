import numpy as np
import pandas as pd

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException


class DspyModelWrapper:
    """MLflow PyFunc wrapper class for Dspy models.

    This wrapper serves two purposes:
        - It stores the Dspy model along with dspy global settings, which are required for seamless
            saving and loading.
        - It provides a `predict` method so that it can be loaded as an MLflow pyfunc, which is
            used at serving time.
    """

    def __init__(self, model, dspy_settings, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.dspy_settings = dspy_settings

    def predict(self, inputs, **kwargs):
        supported_input_types = (np.ndarray, pd.DataFrame, str)
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
            # Return the output as a string for serving simplicity.
            return str(self.model(inputs))

        if isinstance(inputs, str):
            return str(self.model(inputs))
