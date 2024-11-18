"""
This is an example for leveraging MLflow's auto tracing capabilities for Gemini.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

import os

import mlflow

# Turn on auto tracing for Gemini by calling mlflow.gemini.autolog()
mlflow.gemini.autolog()

# Import the SDK and configure your API key.
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Use the generate_content method to generate responses to your prompts.
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("The opposite of hot is")
print(response.text)

# Also leverage the chat feature to conduct multi-turn interactions
chat = model.start_chat(history=[])
response = chat.send_message("In one sentence, explain how a computer works to a young child.")
print(response.text)
response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")
print(response.text)

# Count tokens for your statement
response = model.count_tokens("The quick brown fox jumps over the lazy dog.")
print(response.total_tokens)

# Generate text embeddings for your content
text = "Hello world"
result = genai.embed_content(model="models/text-embedding-004", content=text)
print(result["embedding"])
