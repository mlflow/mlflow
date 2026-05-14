"""Test-case store: EvaluationDataset wrapper for agent_playground.

Every experiment hosts a single regression dataset named
``regression_suite_<experiment_id>`` (see design doc at
``mlflow/internal:docs/projects/agent-playground``). Rows in that
dataset carry one test case each.

Row shape (per ``dataset.merge_records`` input):

    {
        "inputs": {
            "messages": [...],           # conversation prefix
            "record_id": "tc-xxxx",      # our test_case_id; included to
                                         # uniqueify the input hash so
                                         # two cases on the same prefix
                                         # don't collide
        },
        "expectations": {
            "strategy": "assertion" | "judge",
            "rationale_summary": "...",
            "max_turns": 5,
            "assertion": {...} | None,
            "judge": {...} | None,
            "persona": {...} | None,
        },
        "tags": {
            "source_feedback_id": "fb-...",          # initial feedback
            "source_feedback_ids": "fb-...,fb-...",  # additional feedbacks
                                                     # attached on dedup
            "source_trace_id": "tr-...",
            "source_assistant_message_id": "msg-...",
            "rationale_summary": "...",  # duplicated here for cheap
                                         # filtering / coder-mediated dedup
            "promoted": "true" | "false",
        },
    }

Dedup is by ``sha256(json.dumps(inputs, sort_keys=True))``. Including
``record_id`` in ``inputs`` means each test case has a unique input
hash regardless of conversation prefix overlap.

Module is intentionally a thin layer over ``mlflow.genai.datasets``;
no business logic beyond the row-shape contract and CRUD primitives.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    JudgeSpec,
    PersonaSpec,
    TestCaseRow,
    TestSpec,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

_REGRESSION_DATASET_PREFIX = "regression_suite_"
_SOURCE_FEEDBACK_IDS_DELIM = ","


def regression_dataset_name(experiment_id: str) -> str:
    """Convention: one dataset per experiment, named by experiment id."""
    return f"{_REGRESSION_DATASET_PREFIX}{experiment_id}"


def _new_test_case_id() -> str:
    return f"tc-{uuid.uuid4().hex[:8]}"


def get_or_create_regression_dataset(experiment_id: str) -> EvaluationDataset:
    """Resolve the regression dataset for an experiment, creating if absent.

    Idempotent: callers can invoke this on every write without checking
    existence first.
    """
    name = regression_dataset_name(experiment_id)
    try:
        return get_dataset(name=name)
    except MlflowException as exc:
        if exc.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
    return create_dataset(name=name, experiment_id=experiment_id)


def _spec_to_expectations(spec: TestSpec) -> dict[str, Any]:
    return {
        "strategy": spec.strategy,
        "rationale_summary": spec.rationale_summary,
        "max_turns": spec.max_turns,
        "assertion": spec.assertion.model_dump() if spec.assertion else None,
        "judge": spec.judge.model_dump() if spec.judge else None,
        "persona": spec.persona.model_dump(exclude_none=True) if spec.persona else None,
    }


def _expectations_to_spec(expectations: dict[str, Any]) -> TestSpec:
    assertion = expectations.get("assertion")
    judge = expectations.get("judge")
    persona = expectations.get("persona")
    return TestSpec(
        strategy=expectations["strategy"],
        rationale_summary=expectations["rationale_summary"],
        max_turns=expectations.get("max_turns", 5),
        assertion=AssertionSpec(**assertion) if assertion else None,
        judge=JudgeSpec(**judge) if judge else None,
        persona=PersonaSpec(**persona) if persona else None,
    )


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Tolerate to_df()'s round-trip serializing dicts as JSON strings."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return (
            [item for item in decoded if isinstance(item, dict)]
            if isinstance(decoded, list)
            else []
        )
    return []


def _row_to_test_case(row: dict[str, Any]) -> TestCaseRow:
    inputs = _coerce_dict(row.get("inputs"))
    expectations = _coerce_dict(row.get("expectations"))
    tags = _coerce_dict(row.get("tags"))

    test_case_id = inputs.get("record_id") or ""
    feedback_ids_raw = tags.get("source_feedback_ids", "")
    feedback_ids = [s for s in (feedback_ids_raw or "").split(_SOURCE_FEEDBACK_IDS_DELIM) if s]

    return TestCaseRow(
        test_case_id=test_case_id,
        spec=_expectations_to_spec(expectations),
        conversation_messages=_coerce_list_of_dicts(inputs.get("messages")),
        source_feedback_id=tags.get("source_feedback_id") or None,
        source_feedback_ids=feedback_ids,
        source_trace_id=tags.get("source_trace_id") or None,
        source_assistant_message_id=tags.get("source_assistant_message_id") or None,
        promoted=(tags.get("promoted") == "true"),
    )


def _build_record(
    *,
    test_case_id: str,
    spec: TestSpec,
    conversation_messages: list[dict[str, Any]] | None,
    source_feedback_id: str | None,
    source_feedback_ids: list[str],
    source_trace_id: str | None,
    source_assistant_message_id: str | None,
    promoted: bool,
) -> dict[str, Any]:
    return {
        "inputs": {
            "messages": conversation_messages or [],
            "record_id": test_case_id,
        },
        "expectations": _spec_to_expectations(spec),
        "tags": {
            "source_feedback_id": source_feedback_id or "",
            "source_feedback_ids": _SOURCE_FEEDBACK_IDS_DELIM.join(source_feedback_ids),
            "source_trace_id": source_trace_id or "",
            "source_assistant_message_id": source_assistant_message_id or "",
            "rationale_summary": spec.rationale_summary,
            "promoted": "true" if promoted else "false",
        },
    }


def insert_case(
    experiment_id: str,
    spec: TestSpec,
    *,
    conversation_messages: list[dict[str, Any]] | None = None,
    source_feedback_id: str | None = None,
    source_trace_id: str | None = None,
    source_assistant_message_id: str | None = None,
    promoted: bool = False,
    test_case_id: str | None = None,
) -> str:
    """Insert a new test case row.

    Args:
        experiment_id: Owning experiment.
        spec: TestSpec payload.
        conversation_messages: Conversation prefix replayed at run time.
            For single-turn cases this can be empty; the runner uses the
            persona's seed message.
        source_feedback_id: Assessment id that produced this case. Optional
            for headless / scripted inserts.
        source_trace_id: Failing trace anchored by the feedback.
        source_assistant_message_id: Anchored assistant message id; used
            by the hard-match dedup pre-filter.
        promoted: ``True`` once an accepted-fix workflow marks the case
            as part of the regression gate.
        test_case_id: Override the generated id. Mostly for tests.

    Returns:
        The ``test_case_id`` (newly generated unless overridden).
    """
    test_case_id = test_case_id or _new_test_case_id()
    feedback_ids = [source_feedback_id] if source_feedback_id else []
    record = _build_record(
        test_case_id=test_case_id,
        spec=spec,
        conversation_messages=conversation_messages,
        source_feedback_id=source_feedback_id,
        source_feedback_ids=feedback_ids,
        source_trace_id=source_trace_id,
        source_assistant_message_id=source_assistant_message_id,
        promoted=promoted,
    )
    dataset = get_or_create_regression_dataset(experiment_id)
    dataset.merge_records([record])
    return test_case_id


def list_cases(experiment_id: str) -> list[TestCaseRow]:
    """Return every case in the experiment's regression dataset.

    Returns an empty list if the dataset doesn't exist yet (no inserts
    have happened) or has zero rows.
    """
    try:
        dataset = get_dataset(name=regression_dataset_name(experiment_id))
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return []
        raise

    df = dataset.to_df()
    if df.empty:
        return []
    return [_row_to_test_case(row.to_dict()) for _, row in df.iterrows()]


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
    try:
        dataset = get_dataset(name=regression_dataset_name(experiment_id))
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return False
        raise

    record_id = _find_dataset_record_id(dataset, test_case_id)
    if record_id is None:
        return False
    dataset.delete_records([record_id])
    return True


def update_case(
    experiment_id: str,
    test_case_id: str,
    *,
    spec: TestSpec | None = None,
    conversation_messages: list[dict[str, Any]] | None = None,
    promoted: bool | None = None,
    attach_source_feedback_id: str | None = None,
) -> TestCaseRow | None:
    """Apply a partial update to an existing case.

    Implemented as delete-then-insert because ``merge_records`` dedupes
    by ``sha256(inputs)``; mutating ``inputs.messages`` would otherwise
    leave the old row behind alongside the new. All preserved fields
    (lineage tags, ``promoted``) survive the round-trip.

    Args:
        experiment_id: Owning experiment.
        test_case_id: Id of the case to update.
        spec: Replacement ``TestSpec``. ``None`` keeps the existing one.
        conversation_messages: Replacement prefix. ``None`` keeps the
            existing list.
        promoted: Set/unset the promoted flag. ``None`` leaves it alone.
        attach_source_feedback_id: Append a feedback id to
            ``source_feedback_ids`` (used by the coder-mediated dedup
            attach flow). ``None`` is a no-op.

    Returns:
        The updated ``TestCaseRow``, or ``None`` if the case wasn't found.
    """
    existing = get_case(experiment_id, test_case_id)
    if existing is None:
        return None

    next_spec = spec if spec is not None else existing.spec
    next_messages = (
        conversation_messages
        if conversation_messages is not None
        else existing.conversation_messages
    )
    next_promoted = promoted if promoted is not None else existing.promoted
    next_feedback_ids = list(existing.source_feedback_ids)
    if attach_source_feedback_id and attach_source_feedback_id not in next_feedback_ids:
        next_feedback_ids.append(attach_source_feedback_id)

    # Delete then re-insert so the hash-based dedup doesn't leave the
    # old row behind when `inputs` changes.
    delete_case(experiment_id, test_case_id)
    record = _build_record(
        test_case_id=test_case_id,
        spec=next_spec,
        conversation_messages=next_messages,
        source_feedback_id=existing.source_feedback_id,
        source_feedback_ids=next_feedback_ids,
        source_trace_id=existing.source_trace_id,
        source_assistant_message_id=existing.source_assistant_message_id,
        promoted=next_promoted,
    )
    get_or_create_regression_dataset(experiment_id).merge_records([record])
    return get_case(experiment_id, test_case_id)


def list_anchors(experiment_id: str) -> list[tuple[str | None, str]]:
    """Cheap projection for the hard-match dedup pre-filter.

    Returns ``(source_assistant_message_id, test_case_id)`` tuples for
    every case. Order matches ``list_cases``.
    """
    return [
        (case.source_assistant_message_id, case.test_case_id) for case in list_cases(experiment_id)
    ]


def list_summaries(experiment_id: str) -> list[tuple[str, str]]:
    """Cheap projection for the coder-mediated dedup pass.

    Returns ``(rationale_summary, test_case_id)`` tuples for every case.
    """
    return [(case.spec.rationale_summary, case.test_case_id) for case in list_cases(experiment_id)]


__all__ = [
    "delete_case",
    "get_case",
    "get_or_create_regression_dataset",
    "insert_case",
    "list_anchors",
    "list_cases",
    "list_summaries",
    "regression_dataset_name",
    "update_case",
]
