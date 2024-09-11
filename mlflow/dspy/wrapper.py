from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException


class DspyModelWrapper:
    def __init__(self, model, dspy_settings, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.dspy_settings = dspy_settings

    def predict(self, data, **kwargs):
        if not isinstance(data, str):
            raise MlflowException(
                f"`data` must be a string, but received type: {type(data)}.",
                INVALID_PARAMETER_VALUE,
            )
        return self.model(data)
