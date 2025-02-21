from unittest.mock import Mock, patch

import numpy as np
import txtai

import mlflow
from mlflow.entities import SpanType

from tests.tracing.helper import get_traces


def _agent(self, *args, **kwargs):
    def run(s, *a, **kw):
        return "The Roman Empire ruled the Mediterranean and much of Europe"

    self.process = Mock()
    self.process.run = run


def _embeddings_batchsearch(self, *args, **kwargs):
    return [[{"id": 0, "text": "text search result", "score": 1.0}]]


def _llm(self, *args, **kwargs):
    def generator(s, *a, **kw):
        return "The Roman Empire ruled the Mediterranean and much of Europe"

    self.generator = generator


def _vectors(self, *args, **kwargs):
    return np.random.rand(1, 10)


def test_enable_disable_autolog():
    with patch("txtai.Embeddings.batchsearch", _embeddings_batchsearch):
        mlflow.txtai.autolog()

        embeddings = txtai.Embeddings()
        embeddings.search("test query")

        traces = get_traces()
        assert len(traces) == 1

        mlflow.txtai.autolog(disable=True)
        embeddings.search("test query")

        # New trace should not be created
        traces = get_traces()
        assert len(traces) == 1


def test_agent():
    with patch("txtai.Agent.__init__", _agent):
        mlflow.txtai.autolog()

        agent = txtai.Agent()
        response = agent("Tell me about the Roman Empire")

        trace = get_traces()[0]
        assert trace is not None
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 1

        span = trace.data.spans[0]
        assert span.span_type == SpanType.AGENT
        assert span.inputs == {"text": "Tell me about the Roman Empire"}
        assert span.outputs == response


def test_ann():
    mlflow.txtai.autolog()

    ann = txtai.ann.NumPy({})
    ann.backend = np.random.rand(1, 10)

    results = ann.search(np.random.rand(1, 10), 1)

    trace = get_traces()[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1

    span = trace.data.spans[0]
    assert span.span_type == SpanType.RETRIEVER
    assert span.inputs["limit"] == 1
    assert np.allclose(span.outputs[0], results)


def test_database():
    mlflow.txtai.autolog()

    database = txtai.database.SQLite({"content": True})
    database.insert([(0, "test", None)])
    results = database.search("select id, text from txtai")

    trace = get_traces()[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1

    span = trace.data.spans[0]
    assert span.span_type == SpanType.RETRIEVER
    assert span.inputs == {"query": "select id, text from txtai"}
    assert span.outputs == results


def test_embeddings():
    with patch("txtai.Embeddings.batchsearch", _embeddings_batchsearch):
        mlflow.txtai.autolog()

        embeddings = txtai.Embeddings()
        results = embeddings.batchsearch("apple")

        trace = get_traces()[0]
        assert trace is not None
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 1

        span = trace.data.spans[0]
        assert span.span_type == SpanType.RETRIEVER
        assert span.inputs == {"args": ["apple"]}
        assert span.outputs == results


def test_llm():
    with patch("txtai.LLM.__init__", _llm):
        mlflow.txtai.autolog()

        llm = txtai.LLM()
        response = llm("Tell me about the Roman Empire")

        trace = get_traces()[0]
        assert trace is not None
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 1

        span = trace.data.spans[0]
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"text": "Tell me about the Roman Empire"}
        assert span.outputs == response


def test_pipeline():
    mlflow.txtai.autolog()

    segment = txtai.pipeline.Segmentation(lines=True)
    results = segment("abc\ndef")

    trace = get_traces()[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1

    span = trace.data.spans[0]
    assert span.span_type == SpanType.PARSER
    assert span.inputs == {"text": "abc\ndef"}
    assert span.outputs == results


def test_scoring():
    mlflow.txtai.autolog()

    scoring = txtai.scoring.BM25({"terms": True})
    scoring.index(documents=[(0, "test", None)])
    results = scoring.search("test", 1)

    trace = get_traces()[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 2

    span = trace.data.spans[0]
    assert span.span_type == SpanType.RETRIEVER
    assert span.inputs == {"limit": 1, "query": "test"}
    assert np.allclose(span.outputs, results)


def test_vectors():
    with patch("txtai.vectors.Vectors.encode", _vectors):
        mlflow.txtai.autolog()

        vectors = txtai.vectors.Vectors({}, None, None)
        vectors.dimensionality, vectors.qbits = None, None

        results = vectors.vectorize(["test"])

        trace = get_traces()[0]
        assert trace is not None
        assert trace.info.status == "OK"
        assert len(trace.data.spans) == 1

        span = trace.data.spans[0]
        assert span.span_type == SpanType.EMBEDDING
        assert span.inputs == {"data": ["test"]}
        assert str(span.outputs) == str(results)


def test_workflow():
    mlflow.txtai.autolog()

    # No-op workflow that returns inputs
    workflow = txtai.Workflow(tasks=[txtai.workflow.Task()])
    results = list(workflow(["workflow input"]))

    trace = get_traces()[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 2

    # Check workflow
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAIN
    assert span.inputs == {"elements": results}
    assert span.outputs == results

    # Check task
    span = trace.data.spans[1]
    assert span.span_type == SpanType.PARSER
    assert span.inputs["elements"] == results
    assert span.outputs == results
