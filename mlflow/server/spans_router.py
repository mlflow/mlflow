import functools
import os
from typing import Callable, ParamSpec, TypeVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlflow.exceptions import MlflowException
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR, _get_tracking_store

prefix = os.environ.get(STATIC_PREFIX_ENV_VAR, "")
router = APIRouter(prefix=f"{prefix}/api/3.0")


P = ParamSpec("P")
R = TypeVar("R")


def catch_mlflow_exception(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except MlflowException as e:
            raise HTTPException(
                status_code=e.get_http_status_code(), detail=e.serialize_as_json()
            ) from e

    return wrapper


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
@catch_mlflow_exception
async def get_trace_span(
    trace_id: str,
    span_id: str,
) -> GetTraceSpanResponse:
    store = _get_tracking_store()
    span = await store.get_trace_span_async(trace_id, span_id)
    return GetTraceSpanResponse(span=span.to_dict())


class GetTraceSpanContentResponse(BaseModel):
    content: str
    next_page_token: str | None = None


@router.get("/traces/{trace_id}/spans/{span_id}/content")
@catch_mlflow_exception
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
@catch_mlflow_exception
async def list_trace_spans(
    trace_id: str,
    span_type: str | None = None,
    max_results: int | None = None,
    max_content_length: int = 100_000,
    page_token: str | None = None,
) -> ListTraceSpansResponse:
    raise NotImplementedError("TODO: Implement span listing logic")
