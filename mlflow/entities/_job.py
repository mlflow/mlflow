from dataclasses import dataclass
import json
from typing import Any

from mlflow.entities._job_status import JobStatus
from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.jobs_pb2 import JobProgress as ProtoJobProgress
from mlflow.utils.workspace_utils import resolve_entity_workspace_name


@dataclass
class JobProgress:
    """
    Structured best-effort progress payload for an in-flight job.

    This mirrors the RFC's `JobProgress` shape so progress can remain
    machine-readable while still being stored as JSON in the backing row.
    """

    phase: str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobProgress":
        return cls(
            phase=payload.get("phase"),
            completed=payload.get("completed"),
            total=payload.get("total"),
            unit=payload.get("unit"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {}
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.completed is not None:
            payload["completed"] = self.completed
        if self.total is not None:
            payload["total"] = self.total
        if self.unit is not None:
            payload["unit"] = self.unit
        return payload

    @classmethod
    def from_proto(cls, proto: ProtoJobProgress) -> "JobProgress":
        return cls(
            phase=proto.phase or None,
            completed=proto.completed if proto.HasField("completed") else None,
            total=proto.total if proto.HasField("total") else None,
            unit=proto.unit or None,
        )

    def to_proto(self) -> ProtoJobProgress:
        progress = ProtoJobProgress()
        if self.phase is not None:
            progress.phase = self.phase
        if self.completed is not None:
            progress.completed = self.completed
        if self.total is not None:
            progress.total = self.total
        if self.unit is not None:
            progress.unit = self.unit
        return progress


class Job(_MlflowObject):
    """
    MLflow entity representing a Job.
    """

    def __init__(
        self,
        job_id: str,
        creation_time: int,
        job_name: str,
        params: str,
        timeout: float | None,
        status: JobStatus,
        result: str | None,
        retry_count: int,
        last_update_time: int,
        workspace: str | None = None,
        status_details: dict[str, Any] | None = None,
        error_message: str | None = None,
        executor_backend: str | None = None,
        lease_expires_at: int | None = None,
        status_message: str | None = None,
        progress_payload: JobProgress | dict[str, Any] | None = None,
        progress_updated_at: int | None = None,
        token_hash: str | None = None,
        scoped_permissions: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._job_id = job_id
        self._creation_time = creation_time
        self._job_name = job_name
        self._params = params
        self._timeout = timeout
        self._status = status
        self._result = result
        self._retry_count = retry_count
        self._last_update_time = last_update_time
        self._workspace = resolve_entity_workspace_name(workspace)
        self._status_details = status_details
        self._error_message = error_message
        self._executor_backend = executor_backend
        self._lease_expires_at = lease_expires_at
        self._status_message = status_message
        self._progress_payload = (
            JobProgress.from_dict(progress_payload)
            if isinstance(progress_payload, dict)
            else progress_payload
        )
        self._progress_updated_at = progress_updated_at
        self._token_hash = token_hash
        # Keep this as a dict for now. We expect to tighten it into a typed
        # dataclass once the remote-execution permission model is implemented.
        self._scoped_permissions = scoped_permissions

    @property
    def job_id(self) -> str:
        """String containing job ID."""
        return self._job_id

    @property
    def creation_time(self) -> int:
        """Creation timestamp of the job, in number of milliseconds since the UNIX epoch."""
        return self._creation_time

    @property
    def job_name(self) -> str:
        """
        String containing the static job name that uniquely identifies the decorated job function.
        """
        return self._job_name

    @property
    def params(self) -> str:
        """
        String containing the job serialized parameters in JSON format.
        For example, `{"a": 3, "b": 4}` represents two params:
        `a` with value 3 and `b` with value 4.
        """
        return self._params

    @property
    def timeout(self) -> float | None:
        """
        Job execution timeout in seconds.
        """
        return self._timeout

    @property
    def status(self) -> JobStatus:
        """
        One of the values in :py:class:`mlflow.entities._job_status.JobStatus`
        describing the status of the job.
        """
        return self._status

    @property
    def result(self) -> str | None:
        """
        Durable persisted job outcome payload.

        For successful jobs this stores the serialized return value. For existing
        local job flows and compatibility paths it may also contain a legacy
        string error payload, so it is kept separate from `error_message`
        instead of being repurposed as a display-only operator message.
        """
        return self._result

    @property
    def parsed_result(self) -> Any:
        """
        Return the parsed result.
        If job status is SUCCEEDED, the parsed result is the
        job function returned value
        If job status is FAILED, the parsed result is the error string.
        Otherwise, the parsed result is None.
        """
        if self.status == JobStatus.SUCCEEDED:
            return json.loads(self.result)
        return self.result

    @property
    def retry_count(self) -> int:
        """Integer containing the job retry count"""
        return self._retry_count

    @property
    def last_update_time(self) -> int:
        """Last update timestamp of the job, in number of milliseconds since the UNIX epoch."""
        return self._last_update_time

    @property
    def workspace(self) -> str | None:
        """Workspace associated with this job."""
        return self._workspace

    @property
    def status_details(self) -> dict[str, Any] | None:
        """Job status details containing other runtime information."""
        return self._status_details

    @property
    def error_message(self) -> str | None:
        """
        Human-readable terminal error for operators and UI surfaces.

        In the current compatibility model this is a read-only convenience view
        over terminal failure state. New failure writes use `result` as the
        canonical persisted terminal payload, while `error_message` is derived
        from that payload for failed or timed-out jobs. A directly provided
        value still takes precedence for compatibility with existing callers
        that construct `Job` entities in memory.
        """
        if self._error_message is not None:
            return self._error_message
        if self.status in {JobStatus.FAILED, JobStatus.TIMEOUT} and isinstance(self.result, str):
            return self.result
        return None

    @property
    def executor_backend(self) -> str | None:
        """
        Persisted executor backend selected for the job.

        This is framework coordination state used to keep retries, cancellation,
        and recovery pinned to the same backend choice for the lifetime of the
        job row.
        """
        return self._executor_backend

    @property
    def lease_expires_at(self) -> int | None:
        """
        Expiration timestamp for the job's short-lived execution lease.

        This is distinct from terminal outcome fields and exists so recovery
        logic can tell whether a `RUNNING` job still appears healthy.
        """
        return self._lease_expires_at

    @property
    def status_message(self) -> str | None:
        """
        Latest best-effort in-flight status message.

        This is the lightweight plain-text progress channel for operators and
        simple UI surfaces. It exists so jobs can report useful progress text
        even when they do not emit a structured `progress_payload`.
        """
        return self._status_message

    @property
    def progress_payload(self) -> JobProgress | None:
        """
        Latest best-effort structured progress payload.

        This is intentionally distinct from `status_message`: the string message
        is the human-readable fallback, while `progress_payload` is the
        machine-readable form that can carry fields such as phase, completed,
        total, and unit for richer shared progress UIs.
        """
        return self._progress_payload

    @property
    def progress_updated_at(self) -> int | None:
        """
        Timestamp of the latest structured or message-based progress update, in
        milliseconds since the UNIX epoch.
        """
        return self._progress_updated_at

    @property
    def token_hash(self) -> str | None:
        """
        Persisted hash of the remote-execution job token.

        This is internal framework auth state. The plaintext token is never
        stored on the entity or in the database row.
        """
        return self._token_hash

    @property
    def scoped_permissions(self) -> dict[str, Any] | None:
        """
        Persisted permissions scoped to this job's remote-execution contract.

        This is distinct from user-facing status fields and exists so the server
        can authorize only the resources that a remote job attempt is allowed to
        access.
        """
        return self._scoped_permissions

    def __repr__(self) -> str:
        return f"<Job(job_id={self.job_id}, job_name={self.job_name}, workspace={self.workspace})>"
