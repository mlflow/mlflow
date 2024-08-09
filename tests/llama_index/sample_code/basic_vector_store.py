from llama_index.core import Document, VectorStoreIndex

import mlflow

mlflow.llama_index.autolog()

index = VectorStoreIndex.from_documents(documents=[Document.example()])

mlflow.models.set_model(index)
