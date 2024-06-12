# This is an example for logging a Python model as code using the
# mlflow.pyfunc.log_model API by passing in the corresponding model_as_code.py path,
# which avoids serializing the model but rather stores the model code as an artifact.
# The model is set using mlflow.models.set_model in the model code is the model object
# that will be loaded back using mlflow.pyfunc.load_model, which can be used for inference.


import pandas as pd

import mlflow

input_example = ["What is the weather like today?"]

# Specify the path to the model notebook
model_notebook_path = "model_as_code.py"
print(f"Model notebook path: {model_notebook_path}")

print("Logging model as code using Pyfunc log model API")
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model=model_notebook_path,
        artifact_path="ai-model",
        input_example=input_example,
    )

print("Loading model using Pyfunc load model API")
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
output = pyfunc_model.predict(pd.DataFrame(input_example, columns=["input"]))
print(f"Output: {output}")
