from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.insights.models import EvidenceEntry


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
            try:
                if for_issue:
                    validated_entries.append(
                        EvidenceEntry.for_issue(
                            trace_id=entry.get("trace_id", ""), rationale=entry.get("rationale", "")
                        )
                    )
                else:
                    validated_entries.append(
                        EvidenceEntry.for_hypothesis(
                            trace_id=entry.get("trace_id", ""),
                            rationale=entry.get("rationale", ""),
                            supports=entry.get("supports", True),
                        )
                    )
            except Exception as e:
                if isinstance(e, MlflowException):
                    raise
                raise MlflowException.invalid_parameter_value("Invalid evidence entry") from e
        else:
            raise MlflowException.invalid_parameter_value(
                f"Evidence must be a dict or EvidenceEntry, got {type(entry).__name__}"
            )

    return validated_entries


def extract_trace_ids(evidence: list[EvidenceEntry]) -> list[str]:
    """
    Extract unique trace IDs from evidence entries.

    Args:
        evidence: List of EvidenceEntry objects

    Returns:
        List of unique trace IDs preserving order of first occurrence
    """
    seen = set()
    unique_ids = []
    for entry in evidence:
        if entry.trace_id not in seen:
            seen.add(entry.trace_id)
            unique_ids.append(entry.trace_id)
    return unique_ids
