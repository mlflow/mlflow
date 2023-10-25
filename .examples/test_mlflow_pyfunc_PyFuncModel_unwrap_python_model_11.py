# Location: mlflow/mlflow/pyfunc/__init__.py:505
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/pyfunc/__init__.py:505 '])
def test(_):
    import mlflow


    # define a custom model
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return self.my_custom_function(model_input, params)

        def my_custom_function(self, model_input, params=None):
            # do something with the model input
            return 0


    some_input = 1
    # save the model
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=MyModel())

    # load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    print(type(loaded_model))  # <class 'mlflow.pyfunc.model.PyFuncModel'>

    unwrapped_model = loaded_model.unwrap_python_model()
    print(type(unwrapped_model))  # <class '__main__.MyModel'>

    # does not work, only predict() is exposed
    # print(loaded_model.my_custom_function(some_input))

    print(unwrapped_model.my_custom_function(some_input))  # works

    print(loaded_model.predict(some_input))  # works

    # works, but None is needed for context arg
    print(unwrapped_model.predict(None, some_input))


if __name__ == "__main__":
    test()
