"""
Internal job APIs for UI invocation
"""

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mlflow.entities._job import Job as JobEntity
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException

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
    status: JobStatus
    result: Any
    retry_count: int
    last_update_time: int

    @classmethod
    def from_job_entity(cls, job: JobEntity) -> "Job":
        return cls(
            job_id=job.job_id,
            creation_time=job.creation_time,
            function_fullname=job.function_fullname,
            params=json.loads(job.params),
            timeout=job.timeout,
            status=job.status,
            result=job.parsed_result,
            retry_count=job.retry_count,
            last_update_time=job.last_update_time,
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
        function = _load_function(function_fullname)
        job = submit_job(function, payload.params, payload.timeout)
        return Job.from_job_entity(job)
    except MlflowException as e:
        raise HTTPException(
            status_code=e.get_http_status_code(),
            detail=e.message,
        )


class SearchJobPayload(BaseModel):
    function_fullname: str | None = None
    params: dict[str, Any] | None = None
    statuses: list[JobStatus] | None = None


class SearchJobsResponse(BaseModel):
    """
    Pydantic model for job searching response.
    """

    jobs: list[Job]


@job_api_router.post("/search", response_model=SearchJobsResponse)
def search_jobs(payload: SearchJobPayload) -> SearchJobsResponse:
    from mlflow.server.handlers import _get_job_store

    try:
        store = _get_job_store()
        job_results = [
            Job.from_job_entity(job)
            for job in store.list_jobs(
                function_fullname=payload.function_fullname,
                statuses=payload.statuses,
                params=payload.params,
            )
        ]
        return SearchJobsResponse(jobs=job_results)
    except MlflowException as e:
        raise HTTPException(
            status_code=e.get_http_status_code(),
            detail=e.message,
        )
