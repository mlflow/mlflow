import os

import openai

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

mlflow.openai.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    registered_model_name="openai_model",
)

messages = [
    {
        "role": "user",
        "content": "tell me a joke in 50 words",
    }
]

output = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0,
)
print(output)

# We automatically log the model and trace related artifacts
# A model with name `openai_model` is registered, we can load it back as a PyFunc model
model_name = "openai_model"
model_version = 1
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
print(loaded_model.predict("what is the capital of France?"))
