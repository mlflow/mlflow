"""
This is an example for logging a LlamaIndex index to MLflow and loading it back for querying
via specific engine types - query engine, chat engine, and retriever.

For more information about MLflow LlamaIndex integration, see:
https://mlflow.org/docs/latest/llms/llama-index/index.html
"""

import os

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import mlflow

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"

# Configure LLM and Embedding models
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embeddings = OpenAIEmbedding(model="text-embedding-3-large")

# Get sample documents. In practice, you would load documents from various sources, such as local files.
# https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/
documents = [Document.example() for _ in range(10)]

# Create a LlamaIndex index.
index = VectorStoreIndex.from_documents(documents)

# Log the index to MLflow.
mlflow.set_experiment("llama_index")

with mlflow.start_run() as run:
    model_info = mlflow.llama_index.log_model(
        llama_index_model=index,
        name="chat_index",
        # Log the index with chat engine type. This lets you load the index back as a chat engine
        # using `mlflow.pyfunc.load_model()`` API for querying and deploying.
        engine_type="chat",
        # Passing an input example is optional but highly recommended. This allows MLflow to
        # infer the schema of the input and output data.
        input_example="Hi",
    )
    experiment_id = run.info.experiment_id
    run_id = run.info.run_id
print(f"\033[94mIndex is logged to MLflow Run {run_id}\033[0m")

# Load the index back as a chat engine
chat_model = mlflow.pyfunc.load_model(model_info.model_uri)
response1 = chat_model.predict("Hi")
response2 = chat_model.predict("How are you?")

print("\033[94m-------")
print("Loaded the model back as a chat engine:\n")
print(" User > Hi")
print(f"  🤖  > {response1}")
print(" User > How are you?")
print(f"  🤖  > {response2}")
print("\033[0m")

# You can also load the raw index object back using the `mlflow.llama_index.load_model()` API,
# which allows you to create a different engine on top of the index.
loaded_index = mlflow.llama_index.load_model(model_info.model_uri)
query_engine = loaded_index.as_query_engine()
response = query_engine.query("What is the capital of France?")

print("\033[94m-------")
print("Loaded the model back as a query engine:\n")
print(" User > What is the capital of France?")
print(f"  🔍  > {response}")
print("-------\n\033[0m")

print(
    "\033[92m"
    "🚀 Now run `mlflow ui --port 5000` and open MLflow UI to see the logged information, such as "
    "serialized index, global Settings, model signature, dependencies, and more."
)
print(f" - Run URL: http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}")
print("\033[0m")
