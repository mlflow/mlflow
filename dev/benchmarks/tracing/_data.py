import json
import random
import time
import uuid

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import SpanContext

from mlflow.entities.span import Span, SpanType, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import SpanAttributeKey, TraceTagKey
from mlflow.tracing.utils import TraceJSONEncoder

ENV_CHOICES = ["prod", "staging", "dev"]
NAME_PREFIXES = ["agent_run", "qa_chain", "rag_pipeline", "summarizer"]
WEEK_MS = 7 * 24 * 60 * 60 * 1000

SEED_TRACES = 1000
SEED_SPANS_PER_TRACE = 10


def generate_trace_data(
    experiment_id: str,
    num_spans: int,
    rng: random.Random,
) -> tuple[TraceInfo, list[Span]]:
    trace_id = f"tr-{uuid.uuid4().hex}"
    request_time = int(time.time() * 1000) - rng.randint(0, WEEK_MS)
    name_prefix = rng.choice(NAME_PREFIXES)
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=request_time,
        state=rng.choice([TraceState.OK, TraceState.OK, TraceState.OK, TraceState.ERROR]),
        execution_duration=rng.randint(100, 5000),
        tags={
            TraceTagKey.TRACE_NAME: f"{name_prefix}_{trace_id[-4:]}",
            "env": rng.choice(ENV_CHOICES),
        },
    )

    span_types = [SpanType.LLM, SpanType.RETRIEVER, SpanType.TOOL, SpanType.CHAIN]
    base_ns = 1_000_000_000_000
    spans: list[Span] = []

    for i in range(num_spans):
        is_root = i == 0
        span_type = SpanType.AGENT if is_root else rng.choice(span_types)
        parent_id = None if is_root else rng.choice(range(max(0, i - 3), i))

        trace_num = rng.randint(1, 2**63)
        ctx = SpanContext(
            trace_id=trace_num,
            span_id=i + 1,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
            trace_state=trace_api.TraceState(),
        )

        parent_ctx = None
        if parent_id is not None:
            parent_ctx = SpanContext(
                trace_id=trace_num,
                span_id=parent_id + 1,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
                trace_state=trace_api.TraceState(),
            )

        attrs: dict[str, object] = {}
        if is_root:
            attrs[SpanAttributeKey.INPUTS] = json.dumps(
                {"query": "What is ML?"}, cls=TraceJSONEncoder
            )
            attrs[SpanAttributeKey.OUTPUTS] = json.dumps(
                {"response": "ML is..."}, cls=TraceJSONEncoder
            )

        otel_span = OTelReadableSpan(
            name=f"{span_type.lower()}_{i}" if not is_root else "agent_run",
            context=ctx,
            parent=parent_ctx,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_id),
                "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
                **attrs,
            },
            start_time=base_ns + i * 10_000_000,
            end_time=base_ns + i * 10_000_000 + rng.randint(5_000_000, 50_000_000),
            status=trace_api.Status(trace_api.StatusCode.OK),
            resource=_OTelResource.get_empty(),
        )
        spans.append(create_mlflow_span(otel_span, trace_id, span_type))

    return trace_info, spans


def seed_traces(
    store: SqlAlchemyStore,
    experiment_id: str,
    count: int,
    spans_per_trace: int,
) -> list[str]:
    rng = random.Random(123)
    trace_ids: list[str] = []
    for _ in range(count):
        ti, sp = generate_trace_data(experiment_id, spans_per_trace, rng)
        store.start_trace(ti)
        store.log_spans(experiment_id, sp)
        trace_ids.append(ti.trace_id)
    return trace_ids
