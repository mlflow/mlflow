"""
Internal job APIs for UI invocation
"""

import json
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from mlflow.entities._job import Job as JobEntity

job_api_router = APIRouter(prefix="/ajax-api/3.0/jobs", tags=["Job"])


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

    @staticmethod
    def from_job_entity(job: JobEntity):
        return Job(
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
def query_job(job_id: str) -> Job:
    from mlflow.server.jobs import query_job

    job = query_job(job_id)
    return Job.from_job_entity(job)


class SubmitJobPayload(BaseModel):
    function_fullname: str
    params: dict[str, Any]
    timeout: float | None = None


class SubmitJobResponse(BaseModel):
    """
    Pydantic model for submitting job response.
    """

    job_id: str


@job_api_router.post("/", response_model=Job)
def submit_job(payload: SubmitJobPayload) -> Job:
    from mlflow.server.jobs import submit_job
    from mlflow.server.jobs.job_runner import _load_function

    function_fullname = payload.function_fullname
    function = _load_function(function_fullname)
    job = submit_job(function, payload.params, payload.timeout)
    return Job.from_job_entity(job)
