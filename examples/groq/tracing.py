"""
This is an example for leveraging MLflow's auto tracing capabilities for Groq.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

import groq

import mlflow

# Turn on auto tracing for Groq by calling mlflow.groq.autolog()
mlflow.groq.autolog()

client = groq.Groq()

# Use the create method to create new message
message = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of low latency LLMs.",
        }
    ],
)

print(message.choices[0].message.content)
