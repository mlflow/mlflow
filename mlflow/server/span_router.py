import os

from fastapi import APIRouter
from pydantic import BaseModel

from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

router = APIRouter()
prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "")
router = APIRouter(prefix=f"{prefix}/api/2.0", tags=["span_router"])


class ListSpansResponse(BaseModel): ...


class GetSpanResponse(BaseModel): ...


@router.get("/traces/{trace_id}/spans/{span_id}")
async def get_span(trace_id: str, span_id: str) -> GetSpanResponse: ...


@router.get("/traces/{trace_id}/spans")
async def list_spans(
    trace_id: str,
    span_type: str | None = None,
    page_token: str | None = None,
) -> ListSpansResponse: ...
