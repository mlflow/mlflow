import os

import numpy as np
import openai

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema, TensorSpec

assert "OPENAI_API_KEY" in os.environ, " OPENAI_API_KEY environment variable must be set"


print(
    """
# ******************************************************************************
# Text embeddings
# ******************************************************************************
"""
)

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="text-embedding-ada-002",
        task=openai.embeddings,
        name="model",
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["hello", "world"]))


print(
    """
# ******************************************************************************
# Text embeddings with batch_size parameter
# ******************************************************************************
"""
)

with mlflow.start_run():
    mlflow.openai.log_model(
        model="text-embedding-ada-002",
        task=openai.embeddings,
        name="model",
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
            params=ParamSchema([ParamSpec(name="batch_size", dtype="long", default=1024)]),
        ),
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(["hello", "world"], params={"batch_size": 16}))
