"""Telemetry event classes for agent_playground test-case operations.

Each public CRUD or runner operation in this slice is decorated with
``@record_usage_event(SomeEvent)`` so usage shows up in MLflow's existing
telemetry pipeline (see :mod:`mlflow.telemetry.track` and
:class:`mlflow.telemetry.events.Event`).

This module only defines the event classes. Decorators are wired in when
the relevant entry points exist (later PRs). Event names are stable
identifiers from this point on; ``parse`` / ``parse_result`` methods can
be added alongside the decoration stack without breaking existing
records.
"""

from mlflow.telemetry.events import Event


class TestCaseAddedEvent(Event):
    """Emitted when a new test case row is persisted to the regression dataset."""

    name: str = "agent_playground_test_case_added"


class TestCaseUpdatedEvent(Event):
    """Emitted when a test case row is patched (assertion edit, criteria edit, etc.)."""

    name: str = "agent_playground_test_case_updated"


class TestCaseDeletedEvent(Event):
    """Emitted when a test case row is removed."""

    name: str = "agent_playground_test_case_deleted"


class TestRunStartedEvent(Event):
    """Emitted when a regression-suite run begins."""

    name: str = "agent_playground_test_run_started"


class TestRunCompletedEvent(Event):
    """Emitted when a regression-suite run finishes (pass or fail)."""

    name: str = "agent_playground_test_run_completed"


class TestGenJobSubmittedEvent(Event):
    """Emitted when a test-gen job is submitted via ``POST /test-cases/jobs``."""

    name: str = "agent_playground_test_gen_job_submitted"


class TestGenJobCompletedEvent(Event):
    """Emitted when a test-gen job reaches a terminal state."""

    name: str = "agent_playground_test_gen_job_completed"


class PromptFromFeedbackBuiltEvent(Event):
    """Emitted when the fix-prompt endpoint renders a copy-paste prompt."""

    name: str = "agent_playground_prompt_from_feedback_built"


__all__ = [
    "PromptFromFeedbackBuiltEvent",
    "TestCaseAddedEvent",
    "TestCaseDeletedEvent",
    "TestCaseUpdatedEvent",
    "TestGenJobCompletedEvent",
    "TestGenJobSubmittedEvent",
    "TestRunCompletedEvent",
    "TestRunStartedEvent",
]
