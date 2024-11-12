"""
Sample code to define an index using an external vector store (Faiss) for model-from-code logging.

Ref: https://qdrant.tech/documentation/quickstart/
"""

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import mlflow

# Run Qdrant with in-memory mode for testing purpose
client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="test",
    # 1536 is the size of OpenAI embeddings
    vectors_config=VectorParams(size=1536, distance=Distance.DOT),
)
# dummy embeddings
vec = [0.1] * 1536
client.upsert(
    collection_name="test",
    wait=True,
    points=[
        PointStruct(id=1, vector=vec, payload={"text": "hi"}),
        PointStruct(id=1, vector=vec, payload={"text": "hola"}),
    ],
)

vector_store = QdrantVectorStore(client=client, collection_name="test")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


mlflow.models.set_model(index)
