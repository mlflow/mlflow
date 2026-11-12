from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.issues_pb2 import IssueDetectionRun as ProtoIssueDetectionRun


class IssueDetectionRunStatus(str, Enum):
    """Status of an issue detection run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Whether the run has ended, so its status will not change again."""
        return self in (
            IssueDetectionRunStatus.SUCCEEDED,
            IssueDetectionRunStatus.FAILED,
            IssueDetectionRunStatus.CANCELED,
        )


@dataclass
class IssueDetectionRun(_MlflowObject):
    """
    An issue detection run: one analysis of an experiment's traces that files what it finds as
    issues.

    Args:
        run_id: ID of the run. On an MLflow tracking server, issues the run creates carry it as
            their ``source_run_id``. On Databricks, it is the run's Jobs run ID, and issues carry
            ``jobs:<run_id>`` instead.
        experiment_id: The experiment whose traces the run analyzes.
        status: The run's :py:class:`IssueDetectionRunStatus`.
        start_time: When the run started, in milliseconds since the epoch.
        end_time: When the run ended, in milliseconds since the epoch. ``None`` until it ends.
        error_message: Why the run failed. ``None`` unless the status is ``FAILED``.
    """

    run_id: str
    experiment_id: str
    status: IssueDetectionRunStatus
    start_time: int | None = None
    end_time: int | None = None
    error_message: str | None = None

    def to_dictionary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": str(self.status),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dictionary(cls, run_dict: dict[str, Any]) -> IssueDetectionRun:
        return cls(
            run_id=run_dict["run_id"],
            experiment_id=run_dict["experiment_id"],
            status=IssueDetectionRunStatus(run_dict["status"]),
            start_time=run_dict.get("start_time"),
            end_time=run_dict.get("end_time"),
            error_message=run_dict.get("error_message"),
        )

    def to_proto(self) -> ProtoIssueDetectionRun:
        proto = ProtoIssueDetectionRun(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            status=str(self.status),
        )
        if self.start_time is not None:
            proto.start_time = self.start_time
        if self.end_time is not None:
            proto.end_time = self.end_time
        if self.error_message is not None:
            proto.error_message = self.error_message
        return proto

    @classmethod
    def from_proto(cls, proto: ProtoIssueDetectionRun) -> IssueDetectionRun:
        return cls(
            run_id=proto.run_id,
            experiment_id=proto.experiment_id,
            status=IssueDetectionRunStatus(proto.status),
            start_time=proto.start_time if proto.HasField("start_time") else None,
            end_time=proto.end_time if proto.HasField("end_time") else None,
            error_message=proto.error_message if proto.HasField("error_message") else None,
        )
