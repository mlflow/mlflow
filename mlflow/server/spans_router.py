import os

from fastapi import APIRouter
from pydantic import BaseModel

from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "")
router = APIRouter(prefix=f"{prefix}/api/2.0")


class Span(BaseModel):
    trace_id: str
    span_id: str
    # TODO: Add more fields


class ListSpansResponse(BaseModel):
    spans: list[Span]
    next_page_token: str | None = None


class GetSpanResponse(BaseModel):
    span: Span


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_span(trace_id: str, span_id: str) -> GetSpanResponse: ...


@router.get("/traces/{trace_id}/spans")
async def list_spans(
    trace_id: str,
    span_type: str | None = None,
    page_token: str | None = None,
) -> ListSpansResponse: ...
