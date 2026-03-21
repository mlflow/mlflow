from mlflow.models import set_model
from mlflow.pyfunc import PythonModel


class StreamableModel(PythonModel):
    def __init__(self):
        pass

    def predict(self, context, model_input, params=None):
        pass

    def predict_stream(self, context, model_input, params=None):
        yield "test1"
        yield "test2"


set_model(StreamableModel())
