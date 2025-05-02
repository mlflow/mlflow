import os
from pathlib import Path

from promptflow import load_flow

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


# The example flow will write a simple code snippet that displays the greeting message with specific language.
flow_folder = Path(__file__).parent / "basic"
flow = load_flow(flow_folder)

with mlflow.start_run():
    logged_model = mlflow.promptflow.log_model(flow, name="promptflow_model")

loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
print(loaded_model.predict({"text": "Python Hello World!"}))
