"""Deterministic string-based scorers.

These scorers are pure Python comparisons. They cost no tokens, run in
microseconds, and produce stable rationales. Prefer them over LLM judges
whenever the assertion can be expressed as substring / regex / equality.

::

    from mlflow.genai.scorers import Contains, Excludes, Matches, Equals

    Contains("workspace")
    Contains(["register_prompt", "Create Prompt"], match_any=True)
    Contains(["trace", "scorer", "dataset"])         # default: all must match
    Excludes("mlflow runs create")
    Matches(r"mlflow\\.org/docs/.+/genai/prompt")
    Equals("yes", case_sensitive=False)
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from pydantic import Field

from mlflow.entities.assessment import Feedback
from mlflow.genai.scorers.base import Scorer

_NAME_MAX_LEN = 40
_FieldName = Literal["outputs", "inputs"]


def _stringify(value: Any) -> str:
    """Best-effort coercion to string for substring / regex / equality checks.

    Most agent responses are strings already. Dict / structured outputs get
    ``str()``-coerced; users who need precision can extract the field they
    care about in the test body before calling ``verify``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:_NAME_MAX_LEN]


class _StringScorerBase(Scorer):
    """Shared plumbing for the four deterministic string scorers."""

    # Subclasses set this to "contains" / "excludes" / "matches" / "equals".
    KIND: ClassVar[str] = ""

    field: _FieldName = "outputs"
    case_sensitive: bool = False

    def _target_text(self, *, outputs: Any, inputs: Any) -> str:
        return _stringify(outputs if self.field == "outputs" else inputs)


class Contains(_StringScorerBase):
    """Pass iff the targeted field contains the given substring(s).

    By default every substring in ``needles`` must appear (AND). Pass
    ``match_any=True`` for OR semantics.

    Args:
        needles: A single substring or a list of substrings.
        match_any: When ``True`` and a list is given, pass if any substring
            matches. Defaults to ``False`` (all must match).
        case_sensitive: Defaults to ``False``.
        field: Which field to read. ``"outputs"`` (default) or ``"inputs"``.
        name: Display name. Auto-generated from the needles when omitted.
    """

    KIND: ClassVar[str] = "contains"

    needles: list[str] = Field(default_factory=list)
    match_any: bool = False

    def __init__(
        self,
        needles: str | list[str],
        *,
        match_any: bool = False,
        case_sensitive: bool = False,
        field: _FieldName = "outputs",
        name: str | None = None,
    ):
        needle_list = [needles] if isinstance(needles, str) else list(needles)
        if not needle_list:
            raise ValueError("Contains(...) requires at least one substring.")
        super().__init__(
            name=name or _auto_name(self.KIND, needle_list),
            needles=needle_list,
            match_any=match_any,
            case_sensitive=case_sensitive,
            field=field,
        )

    def __call__(self, *, outputs: Any = None, inputs: Any = None) -> Feedback:
        text = self._target_text(outputs=outputs, inputs=inputs)
        haystack = text if self.case_sensitive else text.lower()
        needles = self.needles if self.case_sensitive else [n.lower() for n in self.needles]

        found = [n for n in needles if n in haystack]
        if self.match_any:
            passed = bool(found)
            rationale = (
                f"Found any of {self.needles!r}: {found!r}"
                if passed
                else f"None of {self.needles!r} found in {self.field}."
            )
        else:
            missing = [n for n in needles if n not in haystack]
            passed = not missing
            rationale = (
                f"All required substrings present: {self.needles!r}"
                if passed
                else f"Missing required substrings: {missing!r}"
            )
        return Feedback(name=self.name, value=passed, rationale=rationale)


class Excludes(_StringScorerBase):
    """Pass iff the targeted field contains none of the given substring(s).

    Args:
        needles: A single substring or a list of substrings that must not appear.
        case_sensitive: Defaults to ``False``.
        field: Which field to read. ``"outputs"`` (default) or ``"inputs"``.
        name: Display name. Auto-generated when omitted.
    """

    KIND: ClassVar[str] = "excludes"

    needles: list[str] = Field(default_factory=list)

    def __init__(
        self,
        needles: str | list[str],
        *,
        case_sensitive: bool = False,
        field: _FieldName = "outputs",
        name: str | None = None,
    ):
        needle_list = [needles] if isinstance(needles, str) else list(needles)
        if not needle_list:
            raise ValueError("Excludes(...) requires at least one substring.")
        super().__init__(
            name=name or _auto_name(self.KIND, needle_list),
            needles=needle_list,
            case_sensitive=case_sensitive,
            field=field,
        )

    def __call__(self, *, outputs: Any = None, inputs: Any = None) -> Feedback:
        text = self._target_text(outputs=outputs, inputs=inputs)
        haystack = text if self.case_sensitive else text.lower()
        needles = self.needles if self.case_sensitive else [n.lower() for n in self.needles]

        found = [n for n in needles if n in haystack]
        passed = not found
        rationale = (
            f"No forbidden substrings present in {self.field}."
            if passed
            else f"Forbidden substrings found: {found!r}"
        )
        return Feedback(name=self.name, value=passed, rationale=rationale)


class Matches(_StringScorerBase):
    """Pass iff the targeted field matches the regex pattern.

    Uses ``re.search`` semantics (anywhere in the string, not anchored).
    Use ``^...$`` in the pattern to anchor.

    Args:
        pattern: The regex pattern.
        case_sensitive: Defaults to ``False``. Adds ``re.IGNORECASE`` when ``False``.
        field: Which field to read. ``"outputs"`` (default) or ``"inputs"``.
        name: Display name. Auto-generated when omitted.
    """

    KIND: ClassVar[str] = "matches"

    pattern: str

    def __init__(
        self,
        pattern: str,
        *,
        case_sensitive: bool = False,
        field: _FieldName = "outputs",
        name: str | None = None,
    ):
        if not pattern:
            raise ValueError("Matches(...) requires a non-empty pattern.")
        # Compile-check up front so users get the error at decoration time,
        # not while a test is running.
        flags = 0 if case_sensitive else re.IGNORECASE
        re.compile(pattern, flags=flags)
        super().__init__(
            name=name or _auto_name(self.KIND, [pattern]),
            pattern=pattern,
            case_sensitive=case_sensitive,
            field=field,
        )

    def __call__(self, *, outputs: Any = None, inputs: Any = None) -> Feedback:
        text = self._target_text(outputs=outputs, inputs=inputs)
        flags = 0 if self.case_sensitive else re.IGNORECASE
        match = re.search(self.pattern, text, flags=flags)
        passed = match is not None
        rationale = (
            f"Pattern matched: {match.group(0)!r}"
            if passed
            else f"Pattern {self.pattern!r} not found in {self.field}."
        )
        return Feedback(name=self.name, value=passed, rationale=rationale)


class Equals(_StringScorerBase):
    """Pass iff the targeted field equals the expected value.

    Args:
        expected: The expected string.
        case_sensitive: Defaults to ``False``.
        field: Which field to read. ``"outputs"`` (default) or ``"inputs"``.
        name: Display name. Auto-generated when omitted.
    """

    KIND: ClassVar[str] = "equals"

    expected: str

    def __init__(
        self,
        expected: str,
        *,
        case_sensitive: bool = False,
        field: _FieldName = "outputs",
        name: str | None = None,
    ):
        super().__init__(
            name=name or _auto_name(self.KIND, [expected]),
            expected=expected,
            case_sensitive=case_sensitive,
            field=field,
        )

    def __call__(self, *, outputs: Any = None, inputs: Any = None) -> Feedback:
        text = self._target_text(outputs=outputs, inputs=inputs)
        if self.case_sensitive:
            passed = text == self.expected
        else:
            passed = text.lower() == self.expected.lower()
        rationale = (
            f"Equals {self.expected!r}"
            if passed
            else f"Expected {self.expected!r}, got {text!r}"
        )
        return Feedback(name=self.name, value=passed, rationale=rationale)


def _auto_name(kind: str, needles: list[str]) -> str:
    """Derive a stable, readable scorer name like ``contains_workspace``."""
    if len(needles) == 1:
        return f"{kind}_{_slug(needles[0])}"
    joined = "_".join(_slug(n) for n in needles[:3])
    suffix = "_etc" if len(needles) > 3 else ""
    return f"{kind}_{joined}{suffix}"[:_NAME_MAX_LEN]
