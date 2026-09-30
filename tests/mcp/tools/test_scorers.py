import pytest

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.mcp.tools.scorers import list_scorers, register_llm_judge_scorer


@pytest.fixture
def experiment_id():
    return MlflowClient().create_experiment("exp")


def test_list_builtin_scorers_describes_the_catalog():
    scorers = {s.name: s for s in list_scorers(builtin=True).scorers}
    assert "guidelines" in scorers
    assert scorers["guidelines"].required_args == ["guidelines"]
    assert all(s.session_level is not None for s in scorers.values())


@pytest.mark.parametrize("kwargs", [{}, {"builtin": True, "experiment_id": "0"}])
def test_list_scorers_requires_exactly_one_of_builtin_or_experiment(kwargs):
    with pytest.raises(MlflowException, match="exactly one of builtin or experiment_id"):
        list_scorers(**kwargs)


def test_register_and_list_llm_judge(experiment_id):
    assert list_scorers(experiment_id=experiment_id).scorers == []

    registered = register_llm_judge_scorer(
        name="quality",
        instructions="Is {{ outputs }} a good answer to {{ inputs }}?",
        experiment_id=experiment_id,
        description="Checks quality",
        extra_headers={"X-Team": "a"},
    )
    assert registered.name == "quality"
    assert registered.experiment_id == experiment_id

    (scorer,) = list_scorers(experiment_id=experiment_id).scorers
    assert scorer.name == "quality"
    assert scorer.description == "Checks quality"
    assert scorer.required_args is None


@pytest.mark.parametrize(
    ("extra_headers", "match"),
    [
        ("not json", "invalid JSON"),
        ("[1]", "must be a JSON object"),
        ({"X-Count": 1}, "must all be strings"),
    ],
)
def test_register_llm_judge_validates_extra_headers(experiment_id, extra_headers, match):
    with pytest.raises(MlflowException, match=match):
        register_llm_judge_scorer(
            name="quality",
            instructions="Is {{ outputs }} good?",
            experiment_id=experiment_id,
            extra_headers=extra_headers,
        )
