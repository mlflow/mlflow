"""
Base models and mixins for MLflow Insights.

This module contains the foundational components used by the main entity models.

Developer Primer:
=================

The MLflow Insights system is an AI-guided investigation framework that analyzes
MLflow traces to discover operational and quality issues in AI agents. It follows
a scientific method approach:

1. **Analysis** - A user-initiated investigation run that examines existing traces
2. **Hypothesis** - Testable theories about potential problems discovered during analysis
3. **Evidence** - Trace-based proof that supports or refutes hypotheses
4. **Issue** - Confirmed problems derived from validated hypotheses

Key Components:
---------------
- **EvidenceEntry**: Links specific traces to hypotheses/issues with explanatory rationale
- **SerializableModel**: Provides YAML serialization for artifact storage
- **TimestampedModel**: Tracks creation and modification timestamps
- **ExtensibleModel**: Supports metadata for future extensions
- **EvidencedModel**: Base for entities that collect trace evidence

Evidence Handling:
------------------
Multiple evidence entries can reference the same trace_id. This is intentional and
encouraged as it allows capturing different aspects or insights from the same trace,
providing richer context for the investigation.

Storage Pattern:
----------------
- Analysis: Stored as 'analysis.yaml' in MLflow run artifacts
- Hypotheses: Stored as 'hypothesis_<id>.yaml' in MLflow run artifacts
- Issues: Filed against the experiment (not the run) for visibility across runs

Integration:
------------
Issues can be consumed by coding agents (via CLI/MCP) to automatically fix discovered
problems, enabling a complete feedback loop from issue discovery to resolution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Self

from mlflow.exceptions import MlflowException


class EvidenceEntry(BaseModel):
    """Evidence entry for hypotheses and issues."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    trace_id: str = Field(description="The specific trace ID")
    rationale: str = Field(
        description="Explanation of why this trace supports/refutes the hypothesis or issue"
    )
    supports: bool | None = Field(
        default=None,
        description="Boolean if evidence supports (true) or refutes (false). None for issues.",
    )

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Evidence entry trace_id cannot be empty")
        return v.strip()

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value(
                "Evidence entry rationale cannot be empty"
            )
        return v.strip()

    @classmethod
    def for_hypothesis(
        cls, trace_id: str, rationale: str, supports: bool = True
    ) -> "EvidenceEntry":
        """
        Create an evidence entry for a hypothesis.

        Args:
            trace_id: The trace ID
            rationale: Why this trace is relevant
            supports: Whether evidence supports (True) or refutes (False) the hypothesis

        Returns:
            EvidenceEntry with supports field set
        """
        return cls(trace_id=trace_id, rationale=rationale, supports=supports)

    @classmethod
    def for_issue(cls, trace_id: str, rationale: str) -> "EvidenceEntry":
        """
        Create an evidence entry for an issue.

        Args:
            trace_id: The trace ID
            rationale: Why this trace demonstrates the issue

        Returns:
            EvidenceEntry with supports=None (not applicable for issues)
        """
        return cls(trace_id=trace_id, rationale=rationale, supports=None)


def extract_unique_trace_ids(evidence: list[EvidenceEntry]) -> list[str]:
    """
    Extract unique trace IDs from evidence entries.

    Args:
        evidence: List of EvidenceEntry objects

    Returns:
        List of unique trace IDs preserving order of first occurrence
    """
    return list(dict.fromkeys(entry.trace_id for entry in evidence))


class SerializableModel(BaseModel):
    """Mixin for models that can be serialized to/from YAML."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.safe_dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        """Create instance from YAML string.

        Args:
            yaml_str: YAML formatted string

        Returns:
            Instance of the model class
        """
        data = yaml.safe_load(yaml_str)
        return cls(**data)


class TimestampedModel(BaseModel):
    """Base class for models with timestamp tracking."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp in UTC",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp in UTC",
    )

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current UTC time."""
        self.updated_at = datetime.now(timezone.utc)


class ExtensibleModel(BaseModel):
    """Base class for models with extensible metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible dictionary for custom fields and future extensions",
    )


class EvidencedModel(BaseModel):
    """Base class for models that contain evidence entries.

    Note: Multiple evidence entries with the same trace_id are allowed and encouraged.
    This enables capturing different aspects or rationales from the same trace,
    providing richer context for hypotheses and issues.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    evidence: list[EvidenceEntry] = Field(
        default_factory=list, description="List of evidence entries linking to MLflow traces"
    )

    @property
    def trace_count(self) -> int:
        """Get the number of unique traces associated with this model."""
        return len(extract_unique_trace_ids(self.evidence))

    @property
    def evidence_count(self) -> int:
        """Get the total number of evidence entries."""
        return len(self.evidence)

    def get_trace_ids(self) -> list[str]:
        """Get list of unique trace IDs from evidence."""
        return extract_unique_trace_ids(self.evidence)
