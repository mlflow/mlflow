# This is an example for logging a Python model from code using the
# mlflow.pyfunc.log_model API. When a path to a valid Python script is submitted to the
# python_model argument, the model code itself is serialized instead of the model object.
# Within the targeted script, the model implementation must be defined and set by
# using the mlflow.models.set_model API.

import pandas as pd

import mlflow

input_example = ["What is the weather like today?"]

# Specify the path to the model notebook
model_path = "model_as_code.py"
print(f"Model path: {model_path}")

print("Logging model as code using Pyfunc log model API")
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        name="ai-model",
        input_example=input_example,
    )

print("Loading model using Pyfunc load model API")
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
output = pyfunc_model.predict(pd.DataFrame(input_example, columns=["input"]))
print(f"Output: {output}")
