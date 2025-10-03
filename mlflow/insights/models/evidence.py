from pydantic import BaseModel, Field, field_validator

from mlflow.exceptions import MlflowException


class EvidenceEntry(BaseModel):
    """Evidence entry for hypotheses and issues."""

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
        """Validate trace_id is not empty."""
        if not v or not v.strip():
            raise MlflowException.invalid_parameter_value("Evidence entry trace_id cannot be empty")
        return v.strip()

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        """Validate rationale is not empty."""
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
