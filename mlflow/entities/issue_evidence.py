import json
from dataclasses import dataclass, field

from mlflow.exceptions import MlflowException

ISSUE_EVIDENCE_METADATA_KEY = "mlflow.issue.evidence"
ISSUE_EVIDENCE_SCHEMA_VERSION = 1
# The assessment metadata column is `TEXT`, which MySQL caps at 65,535 bytes. The store writes the
# whole metadata map as one JSON document, so that budget covers every key, every value and the
# escaping the outer encoding adds -- not the evidence payload alone.
MAX_ASSESSMENT_METADATA_BYTES = 65_535
HIGHLIGHT_TONES = ("error", "warning", "success")


def _require_string(value, field_name: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise MlflowException.invalid_parameter_value(f"{field_name} must be a non-empty string")
    return text


@dataclass
class IssueEvidenceStep:
    """
    One step of the walkthrough explaining an issue on a trace.

    Args:
        span_id: The span this step cites. Must belong to the step's trace.
        text: The step's prose, shown in the Issue's evidence card.
        highlights: Exact substrings of ``text`` to emphasize. At least one is required, so a
            step always points at the part of its text that carries the finding.
        highlight_tone: ``error``, ``warning`` or ``success``.
        mono: Whether ``text`` is preformatted, such as a payload or a stack trace.
    """

    span_id: str
    text: str
    highlights: list[str]
    highlight_tone: str = "warning"
    mono: bool = False

    def __post_init__(self):
        self.span_id = _require_string(self.span_id, "span_id")
        self.text = _require_string(self.text, "text")
        if not isinstance(self.highlights, list) or not self.highlights:
            raise MlflowException.invalid_parameter_value(
                "highlights must contain at least one highlight"
            )
        self.highlights = [
            _require_string(highlight, f"highlights[{index}]")
            for index, highlight in enumerate(self.highlights)
        ]
        # Exact substrings keep the rendered emphasis honest: the UI highlights by searching the
        # text, so a paraphrased highlight would silently emphasize nothing.
        if missing := [highlight for highlight in self.highlights if highlight not in self.text]:
            raise MlflowException.invalid_parameter_value(
                f"highlights must be exact substrings of text, but {missing[0]!r} is not"
            )
        if self.highlight_tone not in HIGHLIGHT_TONES:
            raise MlflowException.invalid_parameter_value(
                f"highlight_tone must be one of {', '.join(HIGHLIGHT_TONES)}, "
                f"got {self.highlight_tone!r}"
            )
        if not isinstance(self.mono, bool):
            raise MlflowException.invalid_parameter_value("mono must be a boolean")

    def to_dict(self) -> dict[str, object]:
        step: dict[str, object] = {
            "span_id": self.span_id,
            "text": self.text,
            "highlights": list(self.highlights),
            "highlight_tone": self.highlight_tone,
        }
        if self.mono:
            step["mono"] = True
        return step

    @classmethod
    def from_dict(cls, value, field_name: str = "step") -> "IssueEvidenceStep":
        if not isinstance(value, dict):
            raise MlflowException.invalid_parameter_value(f"{field_name} must be an object")
        mono = value.get("mono", False)
        return cls(
            span_id=value.get("span_id"),
            text=value.get("text"),
            highlights=value.get("highlights"),
            highlight_tone=value.get("highlight_tone", "warning"),
            mono=mono,
        )


@dataclass
class IssueEvidence:
    """
    The evidence for one issue on one trace: a short summary and an ordered walkthrough.

    The summary is stored as the issue reference's rationale and the steps as versioned metadata,
    so a reader that predates structured evidence still shows the summary.

    Args:
        summary: One or two sentences naming what went wrong on this trace.
        steps: The ordered walkthrough. At least one step is required.
        primary_span_id: The span that owns the finding, when one does. Must be cited by a step.
    """

    summary: str
    steps: list[IssueEvidenceStep] = field(default_factory=list)
    primary_span_id: str | None = None

    def __post_init__(self):
        self.summary = _require_string(self.summary, "summary")
        if not isinstance(self.steps, list) or not self.steps:
            raise MlflowException.invalid_parameter_value("evidence must contain at least one step")
        self.steps = [
            step
            if isinstance(step, IssueEvidenceStep)
            else IssueEvidenceStep.from_dict(step, f"steps[{index}]")
            for index, step in enumerate(self.steps)
        ]
        if self.primary_span_id is not None:
            self.primary_span_id = _require_string(self.primary_span_id, "primary_span_id")
            # A primary span nothing walks through would send the trace view to a span the card
            # never explains.
            if all(step.span_id != self.primary_span_id for step in self.steps):
                raise MlflowException.invalid_parameter_value(
                    "primary_span_id must be cited by an evidence step"
                )

    @property
    def span_ids(self) -> list[str]:
        """Every span the walkthrough cites, in step order and without repetition."""
        return list(dict.fromkeys(step.span_id for step in self.steps))

    def validate_spans(self, known_span_ids) -> None:
        """
        Check that every cited span belongs to the trace.

        Args:
            known_span_ids: The span IDs the trace actually carries.

        Raises:
            mlflow.exceptions.MlflowException: If a step cites a span the trace does not contain.
                A fabricated span locator would render an unopenable step, so it is rejected at
                write time.
        """
        known = set(known_span_ids)
        if unknown := [span_id for span_id in self.span_ids if span_id not in known]:
            raise MlflowException.invalid_parameter_value(
                f"Evidence cites span {unknown[0]!r}, which is not part of the trace"
            )

    def to_metadata_value(self) -> str:
        return json.dumps(
            {
                "schema_version": ISSUE_EVIDENCE_SCHEMA_VERSION,
                "steps": [step.to_dict() for step in self.steps],
            },
            separators=(",", ":"),
        )

    def to_metadata(self, metadata: dict[str, str] | None = None) -> dict[str, str]:
        """
        Merge this evidence into an assessment's metadata, bounded by what the store can hold.

        The size is checked on the merged map encoded the way the store encodes it, because that
        is the document the column has to fit: evidence that passes on its own can still overflow
        once the outer encoding escapes it and the caller's own keys are added.

        Raises:
            mlflow.exceptions.MlflowException: If the resulting metadata would not fit in the
                column.
        """
        merged = {**(metadata or {}), ISSUE_EVIDENCE_METADATA_KEY: self.to_metadata_value()}
        stored_bytes = len(json.dumps(merged).encode("utf-8"))
        if stored_bytes > MAX_ASSESSMENT_METADATA_BYTES:
            raise MlflowException.invalid_parameter_value(
                f"Issue reference metadata is {stored_bytes} bytes once stored, which exceeds the "
                f"{MAX_ASSESSMENT_METADATA_BYTES}-byte limit. Shorten the evidence steps."
            )
        return merged

    @classmethod
    def from_metadata_value(cls, summary, value) -> "IssueEvidence | None":
        """
        Rebuild evidence from an issue reference's rationale and metadata.

        Returns ``None`` for a reference that carries no structured evidence, which is how every
        reference written before this schema reads.

        Raises:
            mlflow.exceptions.MlflowException: If the metadata is present but not valid evidence
                of a known schema version.
        """
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            raise MlflowException.invalid_parameter_value("Issue evidence is not valid JSON")
        if not isinstance(parsed, dict):
            raise MlflowException.invalid_parameter_value("Issue evidence must be an object")
        version = parsed.get("schema_version")
        if version != ISSUE_EVIDENCE_SCHEMA_VERSION:
            raise MlflowException.invalid_parameter_value(
                f"Unsupported issue evidence schema version: {version!r}"
            )
        steps = parsed.get("steps")
        if not isinstance(steps, list) or not steps:
            raise MlflowException.invalid_parameter_value(
                "Issue evidence must contain at least one step"
            )
        return cls(summary=summary, steps=steps)

    @classmethod
    def from_assessment(cls, assessment) -> "IssueEvidence | None":
        """
        Read the structured evidence an issue reference assessment carries.

        Returns ``None`` both for a reference that carries no evidence and for one whose stored
        evidence no longer parses. Reads stay lenient on purpose: a reader's job is to show what
        the reference does say, and its rationale remains readable either way. Writes are where
        malformed evidence is rejected.
        """
        metadata = getattr(assessment, "metadata", None) or {}
        summary = getattr(assessment, "rationale", None) or ""
        if not summary:
            return None
        try:
            evidence = cls.from_metadata_value(summary, metadata.get(ISSUE_EVIDENCE_METADATA_KEY))
        except MlflowException:
            return None
        if evidence is None:
            return None
        # The assessment's own span is the primary one, but only when the walkthrough explains it;
        # otherwise the card would offer to open a span it never describes.
        span_id = getattr(assessment, "span_id", None) or None
        if span_id and any(step.span_id == span_id for step in evidence.steps):
            evidence.primary_span_id = span_id
        return evidence
