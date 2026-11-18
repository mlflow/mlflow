import base64
import hashlib
import json
import re
from dataclasses import dataclass

from mlflow.entities.issue import IssueSeverity, IssueStatus
from mlflow.exceptions import MlflowException


def validate_issue_fields(
    *,
    name=None,
    status=None,
    severity=None,
    categories=None,
    metadata=None,
    last_seen_timestamp=None,
    issue_id=None,
    delete_metadata_keys=None,
    created_by=None,
    last_updated_by=None,
):
    if name is not None and (not isinstance(name, str) or not 1 <= len(name) <= 250):
        raise MlflowException.invalid_parameter_value("Issue name must contain 1 to 250 characters")
    for field, value, enum in (
        ("status", status, IssueStatus),
        ("severity", severity, IssueSeverity),
    ):
        if value is not None:
            try:
                enum(value)
            except (ValueError, TypeError):
                raise MlflowException.invalid_parameter_value(f"Invalid issue {field}: {value!r}")
    # bool is an int subtype, but is not a valid timestamp.
    if last_seen_timestamp is not None and (
        type(last_seen_timestamp) is not int or not 0 <= last_seen_timestamp < 2**63
    ):
        raise MlflowException.invalid_parameter_value(
            "last_seen_timestamp must be nonnegative milliseconds"
        )
    if issue_id is not None and (
        not isinstance(issue_id, str) or re.fullmatch(r"iss-[A-Za-z0-9_-]{1,32}", issue_id) is None
    ):
        raise MlflowException.invalid_parameter_value(
            "issue_id must be iss-* and at most 36 characters"
        )
    for field, value, limit in (
        ("created_by", created_by, 255),
        ("last_updated_by", last_updated_by, 255),
    ):
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            raise MlflowException.invalid_parameter_value(f"Invalid {field}")
    for field, values in (
        ("categories", categories),
        ("delete_metadata_keys", delete_metadata_keys),
    ):
        if values is not None and (
            not isinstance(values, list)
            or len(values) > 100
            or any(not isinstance(value, str) or not 1 <= len(value) <= 250 for value in values)
        ):
            raise MlflowException.invalid_parameter_value(
                f"{field} must contain at most 100 nonempty strings of at most 250 characters"
            )
    if metadata is not None and (
        not isinstance(metadata, dict)
        or len(metadata) > 100
        or any(
            not isinstance(key, str)
            or not 1 <= len(key) <= 250
            or not isinstance(value, str)
            or len(value) > 65536
            for key, value in metadata.items()
        )
    ):
        raise MlflowException.invalid_parameter_value(
            "metadata must be a string map with at most 100 entries"
        )
    if set(metadata or {}) & set(delete_metadata_keys or []):
        raise MlflowException.invalid_parameter_value("Cannot update and delete the same key")


def issue_query_fingerprint(**query) -> str:
    return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()


@dataclass
class IssuePagePosition:
    offset: int = 0
    values: list[str | int] | None = None


def issue_page_position(
    token: str | None, fingerprint: str, *, allow_legacy: bool
) -> IssuePagePosition:
    if not token:
        return IssuePagePosition()
    try:
        data = json.loads(base64.b64decode(token, validate=True))
        if not isinstance(data, dict):
            raise ValueError()
        if data.get("query") != fingerprint and not (allow_legacy and "query" not in data):
            raise ValueError()
        if "version" in data:
            values = data.get("after")
            if (
                data["version"] != 2
                or data.get("query") != fingerprint
                or not isinstance(values, list)
                or not 2 <= len(values) <= 101
                or any(
                    not (type(value) is int and -(2**63) <= value < 2**63)
                    and not (isinstance(value, str) and len(value) <= 250)
                    for value in values
                )
            ):
                raise ValueError()
            return IssuePagePosition(values=values)
        offset = data["offset"]
        if type(offset) is not int or offset < 0:
            raise ValueError()
        return IssuePagePosition(offset=offset)
    except (ValueError, TypeError, KeyError):
        raise MlflowException.invalid_parameter_value(
            "Invalid page token or token does not match the issue query"
        )


def issue_keyset_token(values: list[str | int], fingerprint: str) -> str:
    data = {"version": 2, "after": values, "query": fingerprint}
    return base64.b64encode(json.dumps(data).encode()).decode()


def issue_page_token(offset: int, fingerprint: str | None) -> str:
    data = {"offset": offset}
    if fingerprint is not None:
        data["query"] = fingerprint
    return base64.b64encode(json.dumps(data).encode()).decode()


def issue_category_hash(category: str) -> str:
    # A fixed-width key also accommodates unbounded categories stored by older clients.
    return hashlib.sha256(category.strip().lower().encode()).hexdigest()


def validate_issue_search(name, statuses, severities, order_by):
    if name is not None and (not isinstance(name, str) or len(name) > 250):
        raise MlflowException.invalid_parameter_value(
            "name must be a string of at most 250 characters"
        )
    for field, values in (
        ("statuses", statuses),
        ("severities", severities),
        ("order_by", order_by),
    ):
        if values is not None and (
            not isinstance(values, list)
            or len(values) > 100
            or any(not isinstance(value, str) for value in values)
        ):
            raise MlflowException.invalid_parameter_value(
                f"{field} must be a list of at most 100 strings"
            )
