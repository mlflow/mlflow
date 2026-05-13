from typing import Any

from custom_code import iris_classes

import mlflow


class CustomPredict(mlflow.pyfunc.PythonModel):
    """Custom pyfunc class used to create customized mlflow models"""

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        prediction = [x % 3 for x in model_input]
        return iris_classes(prediction)


with mlflow.start_run(run_name="test_custom_model_with_inferred_code_paths"):
    # log a custom model
    model_info = mlflow.pyfunc.log_model(
        name="artifacts",
        infer_code_paths=True,
        python_model=CustomPredict(),
    )
    print(f"Model URI: {model_info.model_uri}")
