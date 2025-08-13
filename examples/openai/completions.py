import os

import openai

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

assert "OPENAI_API_KEY" in os.environ, " OPENAI_API_KEY environment variable must be set"

print(
    """
# ******************************************************************************
# Completions indicating prompt template
# ******************************************************************************
"""
)

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-davinci-002",
        task=openai.completions,
        name="model",
        prompt="Classify the following tweet's sentiment: '{tweet}'.",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["I believe in a better world"]))


print(
    """
# ******************************************************************************
# Completions using inference parameters
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-davinci-002",
        task=openai.completions,
        name="model",
        prompt="Classify the following tweet's sentiment: '{tweet}'.",
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([ColSpec(type="string", name=None)]),
            params=ParamSchema(
                [
                    ParamSpec(name="max_tokens", default=16, dtype="long"),
                    ParamSpec(name="temperature", default=0, dtype="float"),
                    ParamSpec(name="best_of", default=1, dtype="long"),
                ]
            ),
        ),
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["I believe in a better world"], params={"temperature": 1, "best_of": 5}))
