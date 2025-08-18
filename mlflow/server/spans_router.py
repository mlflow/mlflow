import os

from fastapi import APIRouter
from pydantic import BaseModel

from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR, _get_tracking_store

prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "")
router = APIRouter(prefix=f"{prefix}/api/3.0")


class SpanInfo(BaseModel):
    trace_id: str
    span_id: int
    name: str
    span_type: str
    start_time_ms: int
    end_time_ms: int
    duration_ms: int
    status: str


class GetTraceSpanResponse(BaseModel):
    span: SpanInfo


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_trace_span(
    trace_id: str,
    span_id: str,
) -> GetTraceSpanResponse:
    store = _get_tracking_store()
    span = store.get_trace_span(trace_id, span_id)
    return GetTraceSpanResponse(
        span=SpanInfo(
            trace_id=span.trace_id,
            span_id=span.span_id,
            name=span.name,
            span_type=span.span_type,
            start_time_ms=span.start_time_ms,
            end_time_ms=span.end_time_ms,
            duration_ms=span.duration_ms,
            status=span.status,
        )
    )


class GetTraceSpanContentResponse(BaseModel):
    content: str
    next_page_token: str | None = None


@router.get("/traces/{trace_id}/spans/{span_id}/content")
async def get_trace_span_content(
    trace_id: str,
    span_id: str,
    max_content_length: int = 100_000,
    page_token: str | None = None,
) -> GetTraceSpanContentResponse:
    raise NotImplementedError("TODO: Implement span content retrieval logic")


class ListTraceSpansResponse(BaseModel):
    spans: list[SpanInfo]
    next_page_token: str | None = None


@router.get("/traces/{trace_id}/spans")
async def list_trace_spans(
    trace_id: str,
    span_type: str | None = None,
    max_results: int | None = None,
    max_content_length: int = 100_000,
    page_token: str | None = None,
) -> ListTraceSpansResponse:
    raise NotImplementedError("TODO: Implement span listing logic")
