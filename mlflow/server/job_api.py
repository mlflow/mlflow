"""
Internal job APIs for UI invocation
"""
import os
from typing import Any

from fastapi import APIRouter, HTTPException, status as http_status
from pydantic import BaseModel


job_api_router = APIRouter(prefix="/ajax-api/3.0/job", tags=["Job"])


_JOB_FUNCTION_ALLOW_LIST = [
    # Add builtin job functions here
]

if extra_fun_allow_list_env := os.environ.get("_MLFLOW_JOB_FUNCTION_EXTRA_ALLOW_LIST"):
    # this is for injecting allow listed job functions in test
    _JOB_FUNCTION_ALLOW_LIST += extra_fun_allow_list_env.split(",")


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

    function_fullname = payload.function_fullname
    if function_fullname not in _JOB_FUNCTION_ALLOW_LIST:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail=f"The job function {function_fullname} is not in the allowed list.",
        )

    function = _load_function(function_fullname)
    job_id = submit_job(function, payload.params, payload.timeout)
    return SubmitJobResponse(job_id=job_id)
