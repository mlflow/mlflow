from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from mlflow.entities.issue import IssueCategory, IssueSeverity, IssueStatus
from mlflow.utils.annotations import experimental


@experimental(version="3.18.0")
@dataclass
class IssueResource:
    """An experiment-scoped issue returned by the Issues REST API.

    ``name`` is the resource name, such as ``experiments/1/issues/iss-123``;
    ``title`` is its human-readable label. Status and severity use
    :py:class:`mlflow.entities.IssueStatus` and :py:class:`mlflow.entities.IssueSeverity`.
    Built-in categories use :py:class:`mlflow.entities.IssueCategory`; custom categories
    remain strings. Timestamps are RFC 3339 strings.

    Searches using the ``BASIC`` view omit the description and trace-list metadata.
    Use :py:func:`mlflow.genai.get_issue`
    to retrieve the complete issue.
    """

    name: str
    issue_id: str
    title: str
    status: IssueStatus
    severity: IssueSeverity
    description: str | None = None
    categories: list[IssueCategory | str] | None = None
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
        data = {key: value for key, value in asdict(self).items() if value is not None}
        data["status"] = self.status.name
        data["severity"] = self.severity.name
        if self.categories is not None:
            data["categories"] = [
                category.value if isinstance(category, IssueCategory) else category
                for category in self.categories
            ]
        return data

    @classmethod
    def from_dictionary(cls, data: dict[str, Any]) -> IssueResource:
        """Parse the REST representation, allowing additional response fields."""
        values = {field.name: data[field.name] for field in fields(cls) if field.name in data}
        values["status"] = IssueStatus(values["status"].lower())
        values["severity"] = IssueSeverity(values["severity"].lower())
        if values.get("categories") is not None:
            built_in_categories = {category.value: category for category in IssueCategory}
            values["categories"] = [
                built_in_categories.get(category, category) for category in values["categories"]
            ]
        return cls(**values)
