from llama_index.core import Document, VectorStoreIndex

import mlflow
from mlflow.tracing.constant import SpanAttributeKey


def test_autolog_tracing_success():
    mlflow.llama_index.autolog(log_traces=True)

    index = VectorStoreIndex.from_documents([Document.example()])
    res = index.as_query_engine().query("hi")
    assert res

    trace = mlflow.get_last_active_trace()

    assert len(trace.data.spans) == 14
    span = trace.data.spans[0]
    assert span.name == "BaseQueryEngine.query"

    assert span.attributes[SpanAttributeKey.INPUTS]["str_or_query_bundle"] == "hi"
    response = span.attributes[SpanAttributeKey.OUTPUTS]["response"]
    assert response.startswith('[{"role": "system", "content": "You are an expert Q&A')
