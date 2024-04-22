import os

import openai

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

mlflow.openai.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_inputs_outputs=True,
    registered_model_name="openai_model",
)

messages = [
    {
        "role": "user",
        "content": "tell me a joke in 50 words",
    }
]

output = openai.chat.completions.create(
    model="gpt-3.5-turbo-instruct",
    messages=messages,
    temperature=0,
)
print(output)
