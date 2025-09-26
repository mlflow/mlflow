"""
Internal job APIs for UI invocation
"""

import inspect
import json
from typing import Any, Callable

from fastapi import APIRouter
from pydantic import BaseModel

from mlflow.entities._job import Job as JobEntity
from mlflow.exceptions import MlflowException

job_api_router = APIRouter(prefix="/ajax-api/3.0/jobs", tags=["Job"])


def _validate_function_parameters(function: Callable[..., Any], params: dict[str, Any]) -> None:
    """Validate that the provided parameters match the function's required arguments.

    Args:
        function: The function to validate parameters against
        params: Dictionary of parameters provided for the function

    Raises:
        MlflowException: If required parameters are missing
    """
    sig = inspect.signature(function)

    # Get all required parameters (no default value)
    # Exclude VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs) parameters
    required_params = [
        name
        for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        raise MlflowException.invalid_parameter_value(
            f"Missing required parameters for function '{function.__name__}': {missing_params}. "
            f"Expected parameters: {list(sig.parameters.keys())}"
        )


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

    # Validate that required parameters are provided
    _validate_function_parameters(function, payload.params)

    job = submit_job(function, payload.params, payload.timeout)
    return Job.from_job_entity(job)
