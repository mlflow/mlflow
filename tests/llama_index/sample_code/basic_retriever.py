from llama_index.core import Document, VectorStoreIndex

import mlflow

index = VectorStoreIndex.from_documents(documents=[Document.example()])
retriever = index.as_retriever()

mlflow.models.set_model(retriever)
