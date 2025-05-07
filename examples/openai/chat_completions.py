import logging
import os

import openai
import pandas as pd

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

logging.getLogger("mlflow").setLevel(logging.ERROR)

# Uncomment the following lines to run this script without using a real OpenAI API key.
# os.environ["MLFLOW_TESTING"] = "true"
# os.environ["OPENAI_API_KEY"] = "test"

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


print(
    """
# ******************************************************************************
# Single variable
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4o-mini",
        task=openai.chat.completions,
        name="model",
        messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
    )


model = mlflow.pyfunc.load_model(model_info.model_uri)
df = pd.DataFrame(
    {
        "animal": [
            "cats",
            "dogs",
        ]
    }
)
print(model.predict(df))

list_of_dicts = [
    {"animal": "cats"},
    {"animal": "dogs"},
]
print(model.predict(list_of_dicts))

list_of_strings = [
    "cats",
    "dogs",
]
print(model.predict(list_of_strings))
print(
    """
# ******************************************************************************
# Multiple variables
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4o-mini",
        task=openai.chat.completions,
        name="model",
        messages=[{"role": "user", "content": "Tell me a {adjective} joke about {animal}."}],
    )


model = mlflow.pyfunc.load_model(model_info.model_uri)
df = pd.DataFrame(
    {
        "adjective": ["funny", "scary"],
        "animal": ["cats", "dogs"],
    }
)
print(model.predict(df))


list_of_dicts = [
    {"adjective": "funny", "animal": "cats"},
    {"adjective": "scary", "animal": "dogs"},
]
print(model.predict(list_of_dicts))

print(
    """
# ******************************************************************************
# Multiple prompts
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4o-mini",
        task=openai.chat.completions,
        name="model",
        messages=[
            {"role": "system", "content": "You are {person}"},
            {"role": "user", "content": "Let me hear your thoughts on {topic}"},
        ],
    )


model = mlflow.pyfunc.load_model(model_info.model_uri)
df = pd.DataFrame(
    {
        "person": ["Elon Musk", "Jeff Bezos"],
        "topic": ["AI", "ML"],
    }
)
print(model.predict(df))

list_of_dicts = [
    {"person": "Elon Musk", "topic": "AI"},
    {"person": "Jeff Bezos", "topic": "ML"},
]
print(model.predict(list_of_dicts))


print(
    """
# ******************************************************************************
# No input variables
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4o-mini",
        task=openai.chat.completions,
        name="model",
        messages=[{"role": "system", "content": "You are Elon Musk"}],
    )

model = mlflow.pyfunc.load_model(model_info.model_uri)
df = pd.DataFrame(
    {
        "question": [
            "Let me hear your thoughts on AI",
            "Let me hear your thoughts on ML",
        ],
    }
)
print(model.predict(df))

list_of_dicts = [
    {"question": "Let me hear your thoughts on AI"},
    {"question": "Let me hear your thoughts on ML"},
]
model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(list_of_dicts))

list_of_strings = [
    "Let me hear your thoughts on AI",
    "Let me hear your thoughts on ML",
]
model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(list_of_strings))


print(
    """
# ******************************************************************************
# Inference parameters with chat completions
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-4o-mini",
        task=openai.chat.completions,
        name="model",
        messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
        signature=ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([ColSpec(type="string", name=None)]),
            params=ParamSchema(
                [
                    ParamSpec(name="temperature", default=0, dtype="float"),
                ]
            ),
        ),
    )


model = mlflow.pyfunc.load_model(model_info.model_uri)
df = pd.DataFrame(
    {
        "animal": [
            "cats",
            "dogs",
        ]
    }
)
print(model.predict(df, params={"temperature": 1}))
