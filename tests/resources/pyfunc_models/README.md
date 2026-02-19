# Historical Pyfunc Models

These serialized model files are used in backwards compatibility tests, so we can ensure that models logged with old versions of MLflow are still able to be loaded in newer versions.

These files were created by running the following:

1. First, install the desired MLflow version with `$ pip install mlflow=={version_number}`
2. Next, run the following script from MLflow root:

```python
import mlflow


class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input


model = MyModel()

mlflow.pyfunc.save_model(
    python_model=model,
    path=f"tests/resources/pyfunc_models/{mlflow.__version__}",
)
```
