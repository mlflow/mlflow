import argparse
import os

import openai

import mlflow

mlflow.openai.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    registered_model_name="openai_model",
)

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", type=str, help="OpenAI API key")
args = parser.parse_args()
api_key = args.api_key

client = openai.OpenAI(api_key=api_key)

messages = [
    {
        "role": "user",
        "content": "tell me a joke in 50 words",
    }
]

output = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0,
)
print(output)

# We automatically log the model and trace related artifacts
# A model with name `openai_model` is registered, we can load it back as a PyFunc model
# Note that the logged model does not contain any credentials. So, to use the loaded model,
# you will still need to configure the OpenAI API key as an environment variable.
os.environ["OPENAI_API_KEY"] = api_key
model_name = "openai_model"
model_version = 1
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
print(loaded_model.predict("what is the capital of France?"))
