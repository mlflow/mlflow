from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from mlflow.utils.annotations import experimental


@experimental(version="3.18.0")
@dataclass
class IssueResource:
    """An experiment-scoped issue returned by the Issues REST API.

    ``name`` is the resource name, such as ``experiments/1/issues/iss-123``;
    ``title`` is its human-readable label. Enum values use their REST names,
    such as ``PENDING``, ``HIGH``, and ``MANUAL``. Timestamps are RFC 3339 strings.

    Searches using the ``BASIC`` view omit the description and trace-list metadata.
    Use :py:meth:`MlflowClient.get_issue <mlflow.client.MlflowClient.get_issue>`
    to retrieve the complete issue.
    """

    name: str
    issue_id: str
    title: str
    status: str
    severity: str
    description: str | None = None
    categories: list[str] | None = None
    tags: dict[str, str] | None = None
    last_seen_time: str | None = None
    source_type: str = "MANUAL"
    source_id: str | None = None
    created_by: str | None = None
    create_time: str | None = None
    last_updated_by: str | None = None
    update_time: str | None = None
    metadata: dict[str, str] | None = None

    def to_dictionary(self) -> dict[str, Any]:
        """Return the REST representation, omitting fields that are not populated."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dictionary(cls, data: dict[str, Any]) -> IssueResource:
        """Parse the REST representation, allowing additional response fields."""
        return cls(**{field.name: data[field.name] for field in fields(cls) if field.name in data})
