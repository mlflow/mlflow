from mlflow.genai.scorers import Scorer, scorer
from mlflow.genai.scorers.builtin_scorers import get_all_scorers
from mlflow.telemetry.utils import (
    _sanitize_scorer_name,
    is_telemetry_disabled,
    temporarily_disable_telemetry,
)


def test_is_telemetry_disabled(monkeypatch):
    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is False

    with monkeypatch.context() as m:
        m.setenv("DO_NOT_TRACK", "true")
        assert is_telemetry_disabled() is True


def test_temporarily_disable_telemetry(monkeypatch):
    with temporarily_disable_telemetry():
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is False
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    assert is_telemetry_disabled() is True

    with temporarily_disable_telemetry():
        assert is_telemetry_disabled() is True

    assert is_telemetry_disabled() is True


def test_sanitize_scorer_name():
    built_in_scorers = get_all_scorers()
    for built_in_scorer in built_in_scorers:
        assert _sanitize_scorer_name(built_in_scorer) == built_in_scorer.name

    custom_scorer = Scorer(name="test_scorer")
    assert _sanitize_scorer_name(custom_scorer) == "CustomScorer"

    @scorer
    def not_empty(outputs) -> bool:
        return outputs != ""

    assert _sanitize_scorer_name(not_empty) == "CustomScorer"
