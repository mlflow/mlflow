import os
import mlflow

# os.environ["MLFLOW_OPENAI_TESTING"] = "true"
# os.environ["OPENAI_API_KEY"] = "test"

import openai
import pandas as pd


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "user", "content": "Tell me a funny joke about {animal}."}],
    )

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    df = pd.DataFrame(
        {
            "animal": [
                "cats",
                "dogs",
            ],
            "target": [
                "cats",
                "dogs",
            ],
        }
    )
    mlflow.evaluate(
        model=model_info.model_uri,
        data=df,
        targets="target",
        # New model types:
        # - 'summarization'
        # - 'qa'
        # - 'retrieval'
        # - 'text-generation'
        model_type="text-generation",
        evaluators=["text-generation"],
    )


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {
                "role": "system",
                "content": "You're a calculator. Please only give me the answer, which means the response should only contain numbers.",
            },
            {"role": "user", "content": "{x} + {y} ="},
        ],
    )
    model = mlflow.pyfunc.load_model(model_info.model_uri)
    df = pd.DataFrame(
        {
            "x": [
                "1",
                "2",
            ],
            "y": [
                "3",
                "4",
            ],
            "target": [
                "4",
                "6",
            ],
        }
    )
    mlflow.evaluate(
        model=model_info.model_uri,
        data=df,
        targets="target",
        # New model types:
        # - 'summarization'
        # - 'qa'
        # - 'retrieval'
        # - 'text-generation'
        model_type="qa",
        evaluators=["qa"],
    )
