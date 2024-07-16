from llama_index.core import Document, VectorStoreIndex

import mlflow


def test_autolog():
    mlflow.llama_index.autolog(log_traces=True)

    index = VectorStoreIndex.from_documents([Document.example()])
    res = index.as_query_engine().query("hi")
    assert res
