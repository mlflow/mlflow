import pytest

from mlflow.agent_playground.test_cases import telemetry

_ALL_EVENT_CLASSES = [
    telemetry.TestCaseAddedEvent,
    telemetry.TestCaseUpdatedEvent,
    telemetry.TestCaseDeletedEvent,
    telemetry.TestRunStartedEvent,
    telemetry.TestRunCompletedEvent,
    telemetry.PromptFromFeedbackBuiltEvent,
]


@pytest.mark.parametrize("event_cls", _ALL_EVENT_CLASSES)
def test_event_class_has_namespaced_identifier(event_cls):
    assert event_cls.name.startswith("agent_playground_")


def test_event_names_are_unique():
    names = [cls.name for cls in _ALL_EVENT_CLASSES]
    assert len(names) == len(set(names))


def test_events_subclass_mlflow_event_base():
    from mlflow.telemetry.events import Event

    for event_cls in _ALL_EVENT_CLASSES:
        assert issubclass(event_cls, Event)
