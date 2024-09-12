import numpy as np
import pandas as pd

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException


class DspyModelWrapper:
    def __init__(self, model, dspy_settings, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.dspy_settings = dspy_settings

    def predict(self, data, **kwargs):
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(data, np.ndarray):
            flatten = data.reshape(-1)
            if len(flatten) > 1:
                raise MlflowException(
                    "Dspy model doesn't support multiple inputs or batch inference. Please "
                    "provide a single input.",
                    INVALID_PARAMETER_VALUE,
                )
            data = str(flatten[0])
            # Return the output as a string for serving simplicity.
            return str(self.model(data))

        if isinstance(data, str):
            return str(self.model(data))
        supported_input_types = (np.ndarray, pd.DataFrame, str)
        raise MlflowException(
            f"`data` must be one of: {[x.__name__ for x in supported_input_types]}, but "
            f"received type: {type(data)}.",
            INVALID_PARAMETER_VALUE,
        )
