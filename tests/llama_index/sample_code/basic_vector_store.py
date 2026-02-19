from llama_index.core import Document, VectorStoreIndex

import mlflow

index = VectorStoreIndex.from_documents(documents=[Document.example()])

mlflow.models.set_model(index)
