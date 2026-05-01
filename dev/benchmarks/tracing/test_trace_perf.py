"""Performance benchmarks for MLflow tracing.

Run via:
    uv run pytest dev/benchmarks/tracing/ \\
        --benchmark-only \\
        --benchmark-json=benchmark-results.json
"""

import random

from _data import generate_trace_data
from pytest_benchmark.fixture import BenchmarkFixture

import mlflow
from mlflow.entities.span import SpanType
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

DEFAULT_SPANS = 100
INGEST_ROUNDS = 20
INGEST_WARMUP = 3


def test_ingest(benchmark: BenchmarkFixture, store: SqlAlchemyStore, experiment_id: str) -> None:
    rng = random.Random(42)

    def setup():
        ti, sp = generate_trace_data(experiment_id, DEFAULT_SPANS, rng)
        return (ti, sp), {}

    def do(ti, sp):
        store.start_trace(ti)
        store.log_spans(experiment_id, sp)

    benchmark.pedantic(
        do, setup=setup, iterations=1, rounds=INGEST_ROUNDS, warmup_rounds=INGEST_WARMUP
    )


def test_search_by_tag(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="tag.env = 'prod'",
    )


def test_search_by_state(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="status = 'ERROR'",
    )


def test_search_by_name_like(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="name LIKE 'rag_pipeline%'",
    )


def test_search_by_timestamp(
    benchmark: BenchmarkFixture,
    store: SqlAlchemyStore,
    experiment_id: str,
    seeded: list[str],
) -> None:
    benchmark(
        store.search_traces,
        locations=[experiment_id],
        max_results=100,
        filter_string="timestamp > 0",
        order_by=["timestamp DESC"],
    )


def _run_agent_workflow(num_tools: int, num_docs: int, query: str) -> None:
    with mlflow.start_span(name="agent_run", span_type=SpanType.AGENT) as root:
        root.set_inputs({"query": query})

        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as retr:
            retr.set_inputs({"query": query})
            docs = [
                {"id": f"doc_{i}", "score": 0.9 - i * 0.01, "text": f"doc text {i} " * 10}
                for i in range(num_docs)
            ]
            retr.set_outputs({"documents": docs})

        with mlflow.start_span(name="plan", span_type=SpanType.CHAIN) as planner:
            planner.set_inputs({"query": query, "num_docs": len(docs)})
            steps = [f"step_{i}" for i in range(num_tools)]
            planner.set_outputs({"steps": steps})

        tool_results = []
        for step in steps:
            with mlflow.start_span(name=f"tool:{step}", span_type=SpanType.TOOL) as tool:
                tool.set_inputs({"step": step})
                result = {"step": step, "status": "ok", "value": len(step)}
                tool.set_outputs(result)
                tool_results.append(result)

        with mlflow.start_span(name="summarize", span_type=SpanType.LLM) as summ:
            summ.set_inputs({"query": query, "tool_results": tool_results})
            response = f"Answer to {query!r} using {num_docs} docs and {num_tools} tool calls."
            summ.set_outputs({"response": response})
            summ.set_attribute("model", "gpt-test")
            summ.set_attribute("usage.input_tokens", 1234)
            summ.set_attribute("usage.output_tokens", 567)

        root.set_outputs({"response": response})


def test_e2e_agent(benchmark: BenchmarkFixture, e2e_setup: None) -> None:
    counter = [0]

    def do():
        _run_agent_workflow(num_tools=20, num_docs=20, query=f"q-{counter[0]}")
        counter[0] += 1

    benchmark(do)
