import os
import logging

os.environ["OPENAI_API_KEY"] = "653512062c2945ff8476482bb851b2a4"  
os.environ["OPENAI_API_BASE"] = "https://corey-azureopenai.openai.azure.com"  
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_TYPE"] = "azure"

import openai
import mlflow
import pandas as pd
import logging

print(openai.api_base, openai.api_key, openai.api_version, openai.api_type)

logging.getLogger("mlflow").setLevel(logging.ERROR)

print(
    """
# ******************************************************************************
# Single variable
# ******************************************************************************
"""
)
with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        # For Azure OpenAI, model doesn't matter in practice
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
        deployment_id="corey-gpt35",
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
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
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
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
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
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
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
