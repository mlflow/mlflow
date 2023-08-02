import os
import logging

os.environ["OPENAI_API_KEY"] = "<YOUR AZURE OPENAI KEY>"
os.environ["OPENAI_API_BASE"] = "<YOUR AZURE OPENAI BASE>"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_TYPE"] = "azure"

import openai
import mlflow
import pandas as pd
import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        # For Azure OpenAI, model doesn't matter in practice
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
        deployment_id="<YOUR AZURE DEPLOYMENT ID (ALSO CALLED DEPLOYMENT NAME)>",
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

list_of_strings = [
    "Let me hear your thoughts on AI",
    "Let me hear your thoughts on ML",
]
model = mlflow.pyfunc.load_model(model_info.model_uri)
print(model.predict(list_of_strings))
