"""
Internal job APIs for UI invocation
"""

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlflow.entities._job import Job as JobEntity
from mlflow.exceptions import MlflowException

job_api_router = APIRouter(prefix="/ajax-api/3.0/jobs", tags=["Job"])


_ALLOWED_JOB_FUNCTION_LIST = [
    # Putting all allowed job function in the list
]

if allowed_job_function_list_env := os.environ.get("_MLFLOW_ALLOWED_JOB_FUNCTION_LIST"):
    _ALLOWED_JOB_FUNCTION_LIST += allowed_job_function_list_env.split(",")


class Job(BaseModel):
    """
    Pydantic model for job query response.
    """

    job_id: str
    creation_time: int
    function_fullname: str
    params: dict[str, Any]
    timeout: float | None
    status: str
    result: Any
    retry_count: int

    @classmethod
    def from_job_entity(cls, job: JobEntity) -> "Job":
        return cls(
            job_id=job.job_id,
            creation_time=job.creation_time,
            function_fullname=job.function_fullname,
            params=json.loads(job.params),
            timeout=job.timeout,
            status=str(job.status),
            result=job.parsed_result,
            retry_count=job.retry_count,
        )


@job_api_router.get("/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    from mlflow.server.jobs import get_job

    try:
        job = get_job(job_id)
        return Job.from_job_entity(job)
    except MlflowException as e:
        raise HTTPException(
            status_code=e.get_http_status_code(),
            detail=e.message,
        )


class SubmitJobPayload(BaseModel):
    function_fullname: str
    params: dict[str, Any]
    timeout: float | None = None


@job_api_router.post("/", response_model=Job)
def submit_job(payload: SubmitJobPayload) -> Job:
    from mlflow.server.jobs import submit_job
    from mlflow.server.jobs.utils import _load_function

    function_fullname = payload.function_fullname
    try:
        if function_fullname not in _ALLOWED_JOB_FUNCTION_LIST:
            raise MlflowException.invalid_parameter_value(
                f"The function {function_fullname} is not in the allowed job function list"
            )
        function = _load_function(function_fullname)
        job = submit_job(function, payload.params, payload.timeout)
        return Job.from_job_entity(job)
    except MlflowException as e:
        raise HTTPException(
            status_code=e.get_http_status_code(),
            detail=e.message,
        )
