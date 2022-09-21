import mlflow
from mlflow.pyfunc import PythonModel, log_model, load_model


def test_unwrap_python_model_from_pyfunc_class():
    class MyModel(PythonModel):
        def __init__(self, param_1: str, param_2: int):
            self.param_1 = param_1
            self.param_2 = param_2

        def predict(self, context, model_input):
            return model_input + self.param_2

        def upper_param_1(self):
            return self.param_1.upper()

    with mlflow.start_run():
        model = MyModel("this is test message", 2)
        model_uri = log_model(python_model=model, artifact_path="mlruns").model_uri
        loaded_model = load_model(model_uri).unwrap_python_model()
        assert isinstance(loaded_model, MyModel)
        assert loaded_model.param_1 == "this is test message"
        assert loaded_model.param_2 == 2
        assert loaded_model.predict(None, 1) == 3
        assert loaded_model.upper_param_1() == "THIS IS TEST MESSAGE"
