"""
Sample code to define a chat engine and save it with model config (dictionary).
"""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.openai import OpenAI

import mlflow

model_config = mlflow.models.ModelConfig()
model_name = model_config.get("model_name")
temperature = model_config.get("temperature")
llm = OpenAI(model=model_name, temperature=temperature)

index = VectorStoreIndex.from_documents(documents=[Document.example()])
# Setting SIMPLE chat mode will create a SimpleChatEngine instance
chat_engine = index.as_chat_engine(llm=llm, chat_mode=ChatMode.SIMPLE)

mlflow.models.set_model(chat_engine)
