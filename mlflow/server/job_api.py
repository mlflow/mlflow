"""
Internal job APIs for UI invocation
"""

from typing import Any
import json

from fastapi import APIRouter
from mlflow.server.handlers import _get_job_store
from pydantic import BaseModel, Field


job_api_router = APIRouter(prefix="/ajax-api/3.0/job", tags=["Job"])


class QueryJobResponse(BaseModel):
    """
    Pydantic model for job query response.
    """
    status: str
    result: Any


@job_api_router.get("/query/{job_id}", response_model=QueryJobResponse)
def query_job(job_id: str) -> QueryJobResponse:
    from mlflow.server.job import query_job

    status, result = query_job(job_id)
    return QueryJobResponse(
        status=str(status),
        result=result,
    )


class SubmitJobPayload(BaseModel):
    function_fullname: str
    params: dict[str, Any]
    timeout: int | None = None


class SubmitJobResponse(BaseModel):
    """
    Pydantic model for submitting job response.
    """
    job_id: str


@job_api_router.post("/submit", response_model=SubmitJobResponse)
def submit_job(payload: SubmitJobPayload) -> SubmitJobResponse:
    from mlflow.server.job import submit_job
    from mlflow.server.job.job_runner import _load_function

    function = _load_function(payload.function_fullname)
    job = submit_job(function, payload.params, payload.timeout)
    return SubmitJobResponse(job_id=job.job_id)


def _test_fun1(x, y):
    return {
        "a": x + y,
        "b": x - y,
    }
