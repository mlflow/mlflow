"""Test-case store: ``EvaluationDataset`` wrapper for ``agent_playground``.

Every experiment hosts a single regression dataset named
``regression_suite_<experiment_id>`` (see design doc at
``mlflow/internal:docs/projects/agent-playground``). Rows in that
dataset carry one test case each.

The on-disk row shape (per ``dataset.merge_records`` input)::

    {
        "inputs": {
            "messages": [...],            # conversation prefix
            "persona": {...} | None,      # PersonaSpec dump (exclude_none)
            "record_id": "tc-xxxx",       # included in inputs so the
                                          # ``merge_records`` hash uniqueifies
                                          # two cases that share a prefix
        },
        "expectations": {                 # discriminated on ``kind``
            "kind": "assertion" | "judge",
            ...                           # assertion or judge fields
        },
        "tags": {
            "source_feedback_ids": "fb-...,fb-...",  # comma-joined
            "source_trace_id": "tr-...",
            "source_assistant_message_id": "msg-...",
            "promoted": "true" | "false",
            "rationale_summary": "...",
            "max_turns": "5",
        },
    }

In-memory, the row is :class:`TestCaseRow` (defined in ``entities.py``)
which the store layer round-trips via the helpers in this module.

Module is intentionally a thin layer over ``mlflow.genai.datasets``;
no business logic beyond the row-shape contract and CRUD primitives.

Known limitations:

- ``get_or_create_regression_dataset`` has a TOCTOU window between
  ``get_dataset`` and ``create_dataset``. The underlying
  ``SqlEvaluationDataset`` has no UNIQUE constraint on ``name``, so
  two concurrent first-time writers can each create a dataset. If
  that happens, every subsequent ``get_dataset(name=...)`` raises
  ``INVALID_PARAMETER_VALUE`` ("multiple datasets found") until the
  duplicate is removed; :func:`_try_get_regression_dataset` surfaces
  that with a clear message. Mitigation is left to the caller
  (serialize first-time writes per experiment) until the upstream
  constraint lands.
- ``update_case`` is implemented as delete-then-insert.
  ``DatasetRecord`` fields like ``created_time`` and ``source`` are
  reset to "now" on every update. There is a small read-side race
  window: a concurrent ``get_case`` / ``list_cases`` between the
  delete and the re-insert sees the row as gone. Best-effort restore
  on merge failure: if the new merge raises, the original record is
  re-inserted before the exception propagates.
- ``delete_case`` / ``update_case`` are not supported on the
  Databricks tracking backend
  (``EvaluationDataset.delete_records`` raises ``NotImplementedError``
  there). The agent_playground v1 surface targets the local SQL
  backend.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import TypeAdapter, ValidationError

from mlflow.agent_playground.test_cases.entities import (
    Expectations,
    PersonaSpec,
    TestCaseRow,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

_logger = logging.getLogger(__name__)

_REGRESSION_DATASET_PREFIX = "regression_suite_"
_SOURCE_FEEDBACK_IDS_DELIM = ","

# Pre-built ``TypeAdapter`` for the discriminated-union expectations.
# Constructing the adapter once at import time is the pydantic idiom
# for repeated ``validate_python`` calls; doing it per row would
# rebuild the discriminator dispatch on every read.
_EXPECTATIONS_ADAPTER: TypeAdapter[Expectations] = TypeAdapter(Expectations)


def regression_dataset_name(experiment_id: str) -> str:
    """Convention: one dataset per experiment, named by experiment id."""
    return f"{_REGRESSION_DATASET_PREFIX}{experiment_id}"


def new_test_case_id() -> str:
    """Mint a fresh ``tc-<hex>`` id.

    Exposed so callers can construct a :class:`TestCaseRow` with a
    server-style id before calling :func:`insert_case`. 12 hex chars
    (~48 bits) keeps the id short while pushing the birthday-collision
    boundary well past any plausible v1 dataset size; the per-insert
    pre-check in :func:`insert_case` is the final guarantee.
    """
    return f"tc-{uuid.uuid4().hex[:12]}"


def _try_get_regression_dataset(experiment_id: str) -> EvaluationDataset | None:
    """Return the existing regression dataset, or ``None`` if it has not been created yet.

    Surfaces the documented TOCTOU duplicate-name failure mode
    (``INVALID_PARAMETER_VALUE`` from
    ``mlflow.genai.datasets._get_dataset_by_name`` when two concurrent
    first-time writers each created a dataset) with a message that
    names the duplicate and points at the mitigation. Other
    ``MlflowException``\\s propagate with the experiment id chained
    in for debuggability.
    """
    name = regression_dataset_name(experiment_id)
    try:
        return get_dataset(name=name)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        if exc.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE):
            raise MlflowException(
                f"Multiple regression datasets exist for experiment {experiment_id!r} "
                f"under name {name!r}; this indicates a TOCTOU race between two "
                "concurrent first-time writers. Delete the duplicate datasets and "
                "serialize first-time writes per experiment.",
                error_code=exc.error_code,
            ) from exc
        raise MlflowException(
            f"Failed to resolve regression dataset for experiment {experiment_id!r}: {exc.message}",
            error_code=exc.error_code,
        ) from exc


def get_or_create_regression_dataset(experiment_id: str) -> EvaluationDataset:
    """Resolve the regression dataset for an experiment, creating if absent.

    Idempotent: callers can invoke this on every write without checking
    existence first.
    """
    existing = _try_get_regression_dataset(experiment_id)
    if existing is not None:
        return existing
    return create_dataset(name=regression_dataset_name(experiment_id), experiment_id=experiment_id)


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Defensive cast for ``DataFrame`` cells (always dicts in the OSS backend
    where ``SqlEvaluationDatasetRecord.{inputs, expectations, tags}`` are
    ``MutableJSON`` columns, but a corrupted row may carry anything).
    """
    return value if isinstance(value, dict) else {}


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _coerce_int(value: Any, default: int) -> int:
    """Return ``int(value)`` or ``default`` if the value is non-numeric.

    Used for tags written as strings (e.g. ``tags.max_turns`` is
    ``str(int)`` on the wire); a corrupted tag falls back to the
    default instead of taking the whole row down via the malformed-row
    catch in :func:`_row_to_test_case`.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_DEFAULT_MAX_TURNS: int = TestCaseRow.model_fields["max_turns"].default


def _row_to_test_case(row: dict[str, Any]) -> TestCaseRow | None:
    """Decode a stored dataset row into a :class:`TestCaseRow`.

    Returns ``None`` (with a logged warning) when the row is malformed
    so one corrupt row does not poison ``list_cases`` for the whole
    suite. The caller filters ``None`` results out.
    """
    inputs = _coerce_dict(row.get("inputs"))
    expectations_dict = _coerce_dict(row.get("expectations"))
    tags = _coerce_dict(row.get("tags"))

    test_case_id = inputs.get("record_id") or ""
    feedback_ids_raw = tags.get("source_feedback_ids", "")
    feedback_ids = [s for s in feedback_ids_raw.split(_SOURCE_FEEDBACK_IDS_DELIM) if s]
    persona_dict = inputs.get("persona")

    try:
        expectations = _EXPECTATIONS_ADAPTER.validate_python(expectations_dict)
        persona = PersonaSpec(**persona_dict) if isinstance(persona_dict, dict) else None
        return TestCaseRow(
            test_case_id=test_case_id,
            expectations=expectations,
            rationale_summary=tags.get("rationale_summary", ""),
            persona=persona,
            conversation_messages=_coerce_list_of_dicts(inputs.get("messages")),
            max_turns=_coerce_int(tags.get("max_turns"), _DEFAULT_MAX_TURNS),
            source_feedback_ids=feedback_ids,
            source_trace_id=tags.get("source_trace_id") or None,
            source_assistant_message_id=tags.get("source_assistant_message_id") or None,
            promoted=(tags.get("promoted") == "true"),
        )
    except (KeyError, ValueError, ValidationError) as exc:
        _logger.warning(
            "agent_playground store: skipped malformed test-case row %r (%s)",
            test_case_id or "<unknown>",
            exc,
        )
        return None


def _test_case_to_record(case: TestCaseRow) -> dict[str, Any]:
    """Serialize a :class:`TestCaseRow` into the on-disk row shape."""
    return {
        "inputs": {
            "messages": case.conversation_messages,
            "persona": case.persona.model_dump(exclude_none=True) if case.persona else None,
            "record_id": case.test_case_id,
        },
        "expectations": case.expectations.model_dump(),
        "tags": {
            "source_feedback_ids": _SOURCE_FEEDBACK_IDS_DELIM.join(case.source_feedback_ids),
            "source_trace_id": case.source_trace_id or "",
            "source_assistant_message_id": case.source_assistant_message_id or "",
            "promoted": "true" if case.promoted else "false",
            "rationale_summary": case.rationale_summary,
            "max_turns": str(case.max_turns),
        },
    }


def insert_case(experiment_id: str, case: TestCaseRow) -> str:
    """Insert a new test-case row.

    Args:
        experiment_id: Owning experiment.
        case: Fully-validated :class:`TestCaseRow`. Callers that want
            a fresh server-generated id should use
            :func:`new_test_case_id` to mint one before constructing
            the row.

    Returns:
        The ``test_case_id`` (echoed back for caller convenience).

    Raises:
        MlflowException: If a case with the same ``test_case_id``
            already exists in the experiment.
    """
    if get_case(experiment_id, case.test_case_id) is not None:
        raise MlflowException(
            f"Test case {case.test_case_id!r} already exists in experiment {experiment_id!r}",
            error_code=RESOURCE_ALREADY_EXISTS,
        )
    record = _test_case_to_record(case)
    dataset = get_or_create_regression_dataset(experiment_id)
    dataset.merge_records([record])
    return case.test_case_id


def list_cases(experiment_id: str) -> list[TestCaseRow]:
    """Return every case in the experiment's regression dataset.

    Returns an empty list if the dataset doesn't exist yet (no inserts
    have happened) or has zero rows. Malformed rows are logged and
    skipped rather than raising; one bad row does not poison the
    whole list.
    """
    dataset = _try_get_regression_dataset(experiment_id)
    if dataset is None:
        return []

    df = dataset.to_df()
    if df.empty:
        return []
    return [
        case
        for case in (_row_to_test_case(row.to_dict()) for _, row in df.iterrows())
        if case is not None
    ]


def get_case(experiment_id: str, test_case_id: str) -> TestCaseRow | None:
    """Find a single case by id. Returns ``None`` if absent."""
    return next(
        (case for case in list_cases(experiment_id) if case.test_case_id == test_case_id),
        None,
    )


def _find_dataset_record_id(dataset: EvaluationDataset, test_case_id: str) -> str | None:
    df = dataset.to_df()
    if df.empty:
        return None
    for _, row in df.iterrows():
        inputs = _coerce_dict(row.get("inputs"))
        if inputs.get("record_id") == test_case_id:
            return row.get("dataset_record_id")
    return None


def delete_case(experiment_id: str, test_case_id: str) -> bool:
    """Delete a case by id. Returns ``True`` if found and deleted."""
    dataset = _try_get_regression_dataset(experiment_id)
    if dataset is None:
        return False

    record_id = _find_dataset_record_id(dataset, test_case_id)
    if record_id is None:
        return False
    dataset.delete_records([record_id])
    return True


def update_case(
    experiment_id: str,
    test_case_id: str,
    *,
    expectations: Expectations | None = None,
    persona: PersonaSpec | None = None,
    conversation_messages: list[dict[str, Any]] | None = None,
    rationale_summary: str | None = None,
    max_turns: int | None = None,
    promoted: bool | None = None,
    attach_source_feedback_id: str | None = None,
) -> TestCaseRow | None:
    """Apply a partial update to an existing case.

    Implemented as delete-then-insert because ``merge_records`` dedupes
    by ``sha256(inputs)``; mutating ``inputs.messages`` /
    ``inputs.persona`` would otherwise leave the old row behind
    alongside the new. ``DatasetRecord``-level metadata such as
    ``created_time`` and ``source`` is reset to the moment of update.

    ``source_trace_id`` and ``source_assistant_message_id`` are
    deliberately not exposed as kwargs: those tags are one-shot
    lineage capture (the trace + anchored assistant message that
    produced the case) and re-anchoring an existing case to a
    different feedback origin would be confusing. To re-anchor, delete
    and re-insert via :func:`insert_case`.

    On merge failure, the original record is re-inserted (best-effort
    restore) and the original exception re-raises. If the restore
    also fails, an error is logged and the original exception still
    raises.

    Each kwarg defaults to ``None``, meaning "keep the existing
    value". v1 does not support clearing a persona by passing
    ``persona=None`` (since that's the no-op sentinel); to drop a
    persona, delete and re-insert the case.

    Returns:
        The updated :class:`TestCaseRow`, or ``None`` if the case
        wasn't found.
    """
    existing = get_case(experiment_id, test_case_id)
    if existing is None:
        return None

    next_feedback_ids = list(existing.source_feedback_ids)
    if attach_source_feedback_id and attach_source_feedback_id not in next_feedback_ids:
        next_feedback_ids.append(attach_source_feedback_id)

    # ``model_copy(update=...)`` skips validators by design. Round-trip
    # through ``model_validate(model_dump())`` so the strict entity
    # contract is re-checked after the partial update.
    updated = TestCaseRow.model_validate(
        existing.model_copy(
            update={
                "expectations": (
                    expectations if expectations is not None else existing.expectations
                ),
                "persona": persona if persona is not None else existing.persona,
                "conversation_messages": (
                    conversation_messages
                    if conversation_messages is not None
                    else existing.conversation_messages
                ),
                "rationale_summary": (
                    rationale_summary
                    if rationale_summary is not None
                    else existing.rationale_summary
                ),
                "max_turns": max_turns if max_turns is not None else existing.max_turns,
                "promoted": promoted if promoted is not None else existing.promoted,
                "source_feedback_ids": next_feedback_ids,
            }
        ).model_dump()
    )

    existing_record = _test_case_to_record(existing)
    new_record = _test_case_to_record(updated)

    # Delete then re-insert so the hash-based dedup doesn't leave the
    # old row behind when ``inputs`` changes.
    delete_case(experiment_id, test_case_id)
    dataset = get_or_create_regression_dataset(experiment_id)
    try:
        dataset.merge_records([new_record])
    except MlflowException:
        try:
            dataset.merge_records([existing_record])
        except MlflowException:
            _logger.error(
                "agent_playground store: update_case failed to restore record %r "
                "after merge failure; data may be lost",
                test_case_id,
            )
        raise
    return get_case(experiment_id, test_case_id)


def list_anchors(experiment_id: str) -> list[tuple[str | None, str]]:
    """Return ``(source_assistant_message_id, test_case_id)`` per case.

    Used by the hard-match dedup pre-filter. Materializes the full
    dataset under the hood (same cost as ``list_cases``); the
    projection is a convenience shape, not a cheaper path.
    """
    return [
        (case.source_assistant_message_id, case.test_case_id) for case in list_cases(experiment_id)
    ]


def list_summaries(experiment_id: str) -> list[tuple[str, str]]:
    """Return ``(rationale_summary, test_case_id)`` per case.

    Used by the coder-mediated dedup pass. Same materialization cost
    as ``list_cases``.
    """
    return [(case.rationale_summary, case.test_case_id) for case in list_cases(experiment_id)]


__all__ = [
    "delete_case",
    "get_case",
    "get_or_create_regression_dataset",
    "insert_case",
    "list_anchors",
    "list_cases",
    "list_summaries",
    "new_test_case_id",
    "regression_dataset_name",
    "update_case",
]
