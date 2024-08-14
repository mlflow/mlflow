"""
Sample code to define an index using an external vector store (Faiss) for model-from-code logging.

Ref: https://qdrant.tech/documentation/quickstart/
"""
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import mlflow

# Run Qdrant with in-memory mode for testing purpose
client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="test",
    # 1536 is the size of OpenAI embeddings
    vectors_config=VectorParams(size=1536, distance=Distance.DOT),
)

vector_store = QdrantVectorStore(client=client, collection_name="test")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


mlflow.models.set_model(index)
