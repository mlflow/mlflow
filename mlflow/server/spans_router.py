import os

from fastapi import APIRouter
from pydantic import BaseModel

from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "")
router = APIRouter(prefix=f"{prefix}/api/2.0")


class LightWeightSpan(BaseModel):
    trace_id: str
    span_id: int
    name: str
    span_type: str
    start_time_ms: int
    end_time_ms: int
    duration_ms: int
    status: str


class ListTraceSpansResponse(BaseModel):
    spans: list[LightWeightSpan]
    next_page_token: str | None = None


class GetTraceSpanResponse(BaseModel):
    content_slice: str
    next_page_token: str | None = None


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_trace_span(
    trace_id: str,
    span_id: str,
    max_content_length: int = 100_000,
    page_token: str | None = None,
) -> GetTraceSpanResponse:
    raise NotImplementedError("TODO: Implement span retrieval logic")


@router.get("/traces/{trace_id}/spans")
async def list_trace_spans(
    trace_id: str,
    span_type: str | None = None,
    max_content_length: int = 100_000,
    page_token: str | None = None,
) -> ListTraceSpansResponse:
    raise NotImplementedError("TODO: Implement span listing logic")
