"""
Utility functions for MLflow Insights.

This module provides common validation and normalization functions used across
the Insights system for data integrity and consistency.
"""

from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.insights.models.base import EvidenceEntry


def normalize_evidence(
    evidence: list[dict[str, Any]] | list[EvidenceEntry] | None, for_issue: bool = False
) -> list[EvidenceEntry]:
    """
    Normalize evidence entries by converting them to EvidenceEntry objects and leverage type
    and structural validation during the conversion to ensure data integrity in APIs.

    Args:
        evidence: List of evidence as dicts or EvidenceEntry objects
        for_issue: If True, creates entries for issues (supports=None)
                  If False, creates entries for hypotheses (with supports field)

    Returns:
        List of validated EvidenceEntry objects

    Raises:
        MlflowException: If evidence format is invalid
    """
    if not evidence:
        return []

    validated_entries = []

    for entry in evidence:
        if isinstance(entry, EvidenceEntry):
            if for_issue:
                entry.supports = None
            validated_entries.append(entry)

        elif isinstance(entry, dict):
            if "trace_id" not in entry:
                raise MlflowException.invalid_parameter_value(
                    "Evidence entry must include 'trace_id' - trace IDs are required to link "
                    "evidence to specific MLflow traces"
                )
            if "rationale" not in entry:
                raise MlflowException.invalid_parameter_value(
                    "Evidence entry must include 'rationale' - explanation is required for "
                    "why this trace is relevant"
                )

            if for_issue:
                validated_entries.append(
                    EvidenceEntry.for_issue(
                        trace_id=entry["trace_id"], rationale=entry["rationale"]
                    )
                )
            else:
                validated_entries.append(
                    EvidenceEntry.for_hypothesis(
                        trace_id=entry["trace_id"],
                        rationale=entry["rationale"],
                        supports=entry.get("supports", True),
                    )
                )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Evidence must be a dict or EvidenceEntry, got {type(entry).__name__}"
            )

    return validated_entries


def validate_non_empty_string(v: str, field_name: str) -> str:
    """Validate that a string field is not empty or whitespace only.

    Args:
        v: The string value to validate
        field_name: Name of the field for error messages

    Returns:
        The stripped string value

    Raises:
        MlflowException: If the string is empty or whitespace only
    """
    if not v or not v.strip():
        raise MlflowException.invalid_parameter_value(f"{field_name} cannot be empty")
    return v.strip()
