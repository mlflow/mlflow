import os

import openai
import mlflow
import pandas as pd


assert "OPENAI_API_KEY" in os.environ, " OPENAI_API_KEY environment variable must be set"


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-embedding-ada-002",
        task=openai.Embedding,
        artifact_path="model",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
df = pd.DataFrame({"text": ["hello", "world"]})
print(model.predict(df))
