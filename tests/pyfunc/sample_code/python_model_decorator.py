from mlflow.models import set_model
from mlflow.pyfunc import PythonModel


@set_model
class MyModel(PythonModel):
    def predict(self, context, model_input):
        return f"This was the input: {model_input}"


MyModel()
