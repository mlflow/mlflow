from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers.resolution import resolve_scorer_names


def test_resolve_scorer_names_empty():
    assert resolve_scorer_names(set()) == {}


def test_resolve_builtin_scorer_by_name():
    resolved = resolve_scorer_names({"safety"})

    assert set(resolved) == {"safety"}
    assert resolved["safety"].name == "safety"


def test_resolve_multiple_builtin_scorers():
    resolved = resolve_scorer_names({"safety", "correctness"})

    assert set(resolved) == {"safety", "correctness"}


def test_resolve_registered_scorer(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    judge = make_judge(name="my_judge", instructions="Is {{ outputs }} polite?")
    judge.register()

    resolved = resolve_scorer_names({"my_judge"})

    assert resolved["my_judge"].name == "my_judge"


def test_builtin_takes_precedence_over_registry():
    with mock.patch("mlflow.genai.scorers.resolution._get_registered_scorer") as mock_get:
        resolved = resolve_scorer_names({"safety"})

    mock_get.assert_not_called()
    assert resolved["safety"].name == "safety"


def test_resolve_unknown_scorer_name_raises():
    with pytest.raises(MlflowException, match=r"could not be resolved: \['not_a_scorer'\]"):
        resolve_scorer_names({"not_a_scorer"})


def test_error_lists_every_unresolved_name():
    with pytest.raises(MlflowException, match=r"could not be resolved: \['alpha', 'beta'\]"):
        resolve_scorer_names({"beta", "alpha", "safety"})


def test_error_suggests_registering_a_custom_scorer():
    with pytest.raises(MlflowException, match=r"my_scorer\.register\(\)"):
        resolve_scorer_names({"nope"})
