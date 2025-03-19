"""
This is an example for leveraging MLflow's auto tracing capabilities for Gemini.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

import os

import mlflow

# Turn on auto tracing for Gemini by calling mlflow.gemini.autolog()
mlflow.gemini.autolog()

# Import the SDK and configure your API key.
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Use the generate_content method to generate responses to your prompts.
response = client.models.generate_content(
    model="gemini-1.5-flash", contents="The opposite of hot is"
)
print(response.text)

# Also leverage the chat feature to conduct multi-turn interactions
chat = client.chats.create(model="gemini-1.5-flash")
response = chat.send_message("In one sentence, explain how a computer works to a young child.")
print(response.text)
response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")
print(response.text)

# Count tokens for your statement
response = client.models.count_tokens("The quick brown fox jumps over the lazy dog.")
print(response.total_tokens)

# Generate text embeddings for your content
text = "Hello world"
result = client.models.embed_content(model="text-embedding-004", contents=text)
print(result["embedding"])
