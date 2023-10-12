import os

import openai

import mlflow

assert "OPENAI_API_KEY" in os.environ, " OPENAI_API_KEY environment variable must be set"


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-davinci-002",
        task=openai.Completion,
        artifact_path="model",
        prompt="Clasify the following tweet's sentiment: '{tweet}'.",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["I believe in a better world"]))
