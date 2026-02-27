"""
Sample code to define a chat engine and save it with model-from-code logging.

Ref: https://qdrant.tech/documentation/quickstart/
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode

import mlflow

index = VectorStoreIndex.from_documents(documents=[Document.example()])
# Setting SIMPLE chat mode will create a SimpleChatEngine instance
chat_engine = index.as_chat_engine(chat_mode=ChatMode.SIMPLE)

mlflow.models.set_model(chat_engine)
