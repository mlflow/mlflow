from mlflow.models import set_model
from mlflow.pyfunc import PythonModel


class MyModel(PythonModel):
    def predict(self, context=None, model_input=None):
        return f"Predict called with context {context} and input {model_input}"


set_model(MyModel())
