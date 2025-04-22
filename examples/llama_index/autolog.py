"""
This is an example for leveraging MLflow's autologging capabilities for LlamaIndex.

For more information about MLflow LlamaIndex integration, see:
https://mlflow.org/docs/latest/llms/llama-index/index.html
"""

import os

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"

experiment_id = mlflow.set_experiment("llama_index").experiment_id

# Configure LLM
Settings.llm = OpenAI(model="gpt-4o", temperature=0)

# Create a sample LlamaIndex index
documents = [Document.example() for _ in range(10)]
index = VectorStoreIndex.from_documents(documents)

# Turn on autologging
mlflow.llama_index.autolog()

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")
print("\033\n[94m-------")
print("Running Query Engine:\n")
print(" User > What is the capital of France?")
print(f"  🔍  > {response}")

# Interact with the index as a chat engine with streaming API
chat_engine = index.as_chat_engine()
response1 = chat_engine.stream_chat("Hi")
response2 = chat_engine.stream_chat("How are you?")

print("\033\n[94m-------")
print("Running Chat engine:\n")
print(" User > Hi")
print("  🤖  > ", end="")
response1.print_response_stream()
print("\n User > How are you?")
print("  🤖  > ", end="")
response2.print_response_stream()
print("\033[0m")


# Create OpenAI agent
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
agent = OpenAIAgent.from_tools([multiply_tool, add_tool])
response = agent.chat("What is 2 times 3?")
print("\033\n[94m-------")
print("Running Agent:\n")
print(" User > What is 2 times 3?")
print(f"  🦙  > {response}")
print("\n-------\n\n\033[0m")

print("\033[92m🚀 Now run `mlflow ui --port 5000` open MLflow UI to see the trace visualization!")
print(f"   - Experiment URL: http://127.0.0.1:5000/#/experiments/{experiment_id}\033[0m")
