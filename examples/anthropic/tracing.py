"""
This is an example for leveraging MLflow's auto tracing capabilities for Anthropic.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

import os

import mlflow

# Turn on auto tracing for Anthropic by calling mlflow.anthropic.autolog()
mlflow.anthropic.autolog()

# Import the SDK and configure your API key.
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Use the create method to create new message.
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"},
    ],
)
print(message.content)
