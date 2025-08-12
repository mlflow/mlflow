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


class ListSpansResponse(BaseModel):
    spans: list[LightWeightSpan]
    max_content_length: int
    next_page_token: str | None = None


class GetTraceSpanResponse(BaseModel):
    content_slice: str
    max_content_length: int
    next_page_token: str | None = None


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_trace_span(trace_id: str, span_id: str) -> GetTraceSpanResponse:
    raise NotImplementedError("TODO: Implement span retrieval logic")


@router.get("/traces/{trace_id}/spans")
async def list_spans(
    trace_id: str,
    span_type: str | None = None,
    page_token: str | None = None,
) -> ListSpansResponse:
    raise NotImplementedError("TODO: Implement span listing logic")
