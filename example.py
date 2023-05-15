import os
import mlflow

os.environ["MLFLOW_OPENAI_TESTING"] = "true"
os.environ["OPENAI_API_KEY"] = "test"

import openai
import pandas as pd


with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
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
    # This should logs the following using `mlflow.llm.log_predictions`:
    # - Inputs
    # - Outputs
    # - Metrics (e.g. toxicity for text-generation)
