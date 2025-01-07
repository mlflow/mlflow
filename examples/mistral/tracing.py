"""
This is an example for leveraging MLflow's auto tracing capabilities for Mistral AI.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

import os

from mistralai import Mistral

import mlflow

# Turn on auto tracing for Mistral AI by calling mlflow.mistral.autolog()
mlflow.mistral.autolog()

# Configure your API key.
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Use the chat complete method to create new chat.
chat_response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "role": "user",
            "content": "Who is the best French painter? Answer in one short sentence.",
        },
    ],
)
print(chat_response.choices[0].message)
