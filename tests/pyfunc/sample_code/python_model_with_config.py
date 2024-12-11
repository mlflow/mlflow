from mlflow.models import ModelConfig, set_model
from mlflow.pyfunc import PythonModel

base_config = ModelConfig(development_config="tests/pyfunc/sample_code/config.yml")


class MyModel(PythonModel):
    def predict(self, context, model_input):
        timeout = base_config.get("timeout")
        return f"Predict called with input {model_input}, timeout {timeout}"


set_model(MyModel())
