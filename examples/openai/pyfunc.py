import os
import logging

import openai
import mlflow
import pandas as pd
import logging

logging.getLogger("mlflow").setLevel(logging.DEBUG)

# On Databricks, set the stored OpenAI API key scope here for automatically loading the API key
# for real time inference. See https://docs.databricks.com/security/secrets/index.html on
# how to add a scope and API key.
os.environ["MLFLOW_OPENAI_SECRET_SCOPE"] = "<scope-name>"

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "system", "content": "You are an MLflow expert!"}],
    )


print(mlflow.openai.load_model(model_info.model_uri))
# {
#     "messages": [{"content": "You are an MLflow expert!", "role": "system"}],
#     "model": "gpt-3.5-turbo",
#     "task": "chat.completions",
# }

df = pd.DataFrame(
    {
        "role": ["user"] * 10,
        "content": [
            "What is MLflow?",
            "What are the key components of MLflow?",
            "How does MLflow enable reproducibility?",
            "What is MLflow tracking and how does it help?",
            "How can you compare different ML models using MLflow?",
            "How can you use MLflow to deploy ML models?",
            "What are the integrations of MLflow with popular ML libraries?",
            "How can you use MLflow to automate ML workflows?",
            "What security and compliance features does MLflow offer?",
            "Where does MLflow stand in the ML ecosystem?",
        ],
    }
)
model = mlflow.pyfunc.load_model(model_info.model_uri)
print(df.assign(answer=model.predict(df)))
