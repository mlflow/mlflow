"""
Sample code to define an index using an external vector store (Faiss) for model-from-code logging.

Ref: https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/
"""
import os

import faiss
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

import mlflow

FAISS_EMBEDDING_DIM = 1536
FAISS_PERSIST_DIR = "./faiss_persist_dir"

# Only create the index once
if os.path.exists(FAISS_PERSIST_DIR):
    vector_store = FaissVectorStore.from_persist_dir(FAISS_PERSIST_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=FAISS_PERSIST_DIR,
    )
    index = load_index_from_storage(storage_context=storage_context)
else:
    faiss_index = faiss.IndexFlatL2(FAISS_EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=[Document.example(), Document.example()],
        storage_context=storage_context,
    )
    index.storage_context.persist(FAISS_PERSIST_DIR)


mlflow.models.set_model(index)
