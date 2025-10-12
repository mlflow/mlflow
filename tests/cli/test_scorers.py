import json

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli.scorers import commands
from mlflow.genai.scorers import scorer
from mlflow.utils.string_utils import _create_table


@pytest.fixture
def runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture
def experiment():
    """Create a test experiment."""
    experiment_id = mlflow.create_experiment(
        f"test_scorers_cli_{mlflow.utils.time.get_current_time_millis()}"
    )
    yield experiment_id
    mlflow.delete_experiment(experiment_id)


@pytest.fixture
def correctness_scorer():
    """Create a correctness scorer."""

    @scorer
    def _correctness_scorer(outputs) -> bool:
        return len(outputs) > 0

    return _correctness_scorer


@pytest.fixture
def safety_scorer():
    """Create a safety scorer."""

    @scorer
    def _safety_scorer(outputs) -> bool:
        return len(outputs) > 0

    return _safety_scorer


@pytest.fixture
def relevance_scorer():
    """Create a relevance scorer."""

    @scorer
    def _relevance_scorer(outputs) -> bool:
        return len(outputs) > 0

    return _relevance_scorer


@pytest.fixture
def generic_scorer():
    """Create a generic test scorer."""

    @scorer
    def _generic_scorer(outputs) -> bool:
        return True

    return _generic_scorer


def test_commands_group_exists():
    assert commands.name == "scorers"
    assert commands.help is not None


def test_list_command_params():
    list_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "list"), None)
    assert list_cmd is not None
    param_names = {p.name for p in list_cmd.params}
    assert param_names == {"experiment_id", "output"}


def test_list_scorers_table_output(
    runner, experiment, correctness_scorer, safety_scorer, relevance_scorer
):
    correctness_scorer.register(experiment_id=experiment, name="Correctness")
    safety_scorer.register(experiment_id=experiment, name="Safety")
    relevance_scorer.register(experiment_id=experiment, name="RelevanceToQuery")

    result = runner.invoke(commands, ["list", "--experiment-id", experiment])

    assert result.exit_code == 0

    # Construct expected table output (scorers are returned in alphabetical order)
    # Note: click.echo() adds a trailing newline
    expected_table = (
        _create_table([["Correctness"], ["RelevanceToQuery"], ["Safety"]], headers=["Scorer Name"])
        + "\n"
    )
    assert result.output == expected_table


def test_list_scorers_json_output(
    runner, experiment, correctness_scorer, safety_scorer, relevance_scorer
):
    correctness_scorer.register(experiment_id=experiment, name="Correctness")
    safety_scorer.register(experiment_id=experiment, name="Safety")
    relevance_scorer.register(experiment_id=experiment, name="RelevanceToQuery")

    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--output", "json"])

    assert result.exit_code == 0
    output_json = json.loads(result.output)
    assert set(output_json["scorers"]) == {"Correctness", "Safety", "RelevanceToQuery"}


@pytest.mark.parametrize(
    ("output_format", "expected_output"),
    [
        ("table", ""),
        ("json", {"scorers": []}),
    ],
)
def test_list_scorers_empty_experiment(runner, experiment, output_format, expected_output):
    args = ["list", "--experiment-id", experiment]
    if output_format == "json":
        args.extend(["--output", "json"])

    result = runner.invoke(commands, args)
    assert result.exit_code == 0

    if output_format == "json":
        output_json = json.loads(result.output)
        assert output_json == expected_output
    else:
        # Empty table produces minimal output
        assert result.output.strip() == expected_output


def test_list_scorers_with_experiment_id_env_var(runner, experiment, correctness_scorer):
    correctness_scorer.register(experiment_id=experiment, name="Correctness")

    result = runner.invoke(commands, ["list"], env={"MLFLOW_EXPERIMENT_ID": experiment})

    assert result.exit_code == 0
    assert "Correctness" in result.output


def test_list_scorers_missing_experiment_id(runner):
    result = runner.invoke(commands, ["list"])

    assert result.exit_code != 0
    assert "experiment-id" in result.output.lower() or "experiment_id" in result.output.lower()


def test_list_scorers_invalid_output_format(runner, experiment):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--output", "invalid"])

    assert result.exit_code != 0
    assert "invalid" in result.output.lower() or "choice" in result.output.lower()


def test_list_scorers_special_characters_in_names(runner, experiment, generic_scorer):
    generic_scorer.register(experiment_id=experiment, name="Scorer With Spaces")
    generic_scorer.register(experiment_id=experiment, name="Scorer.With.Dots")
    generic_scorer.register(experiment_id=experiment, name="Scorer-With-Dashes")
    generic_scorer.register(experiment_id=experiment, name="Scorer_With_Underscores")

    result = runner.invoke(commands, ["list", "--experiment-id", experiment])

    assert result.exit_code == 0
    assert "Scorer With Spaces" in result.output
    assert "Scorer.With.Dots" in result.output
    assert "Scorer-With-Dashes" in result.output
    assert "Scorer_With_Underscores" in result.output


@pytest.mark.parametrize(
    "output_format",
    ["table", "json"],
)
def test_list_scorers_single_scorer(runner, experiment, generic_scorer, output_format):
    generic_scorer.register(experiment_id=experiment, name="OnlyScorer")

    args = ["list", "--experiment-id", experiment]
    if output_format == "json":
        args.extend(["--output", "json"])

    result = runner.invoke(commands, args)
    assert result.exit_code == 0

    if output_format == "json":
        output_json = json.loads(result.output)
        assert output_json == {"scorers": ["OnlyScorer"]}
    else:
        assert "OnlyScorer" in result.output


@pytest.mark.parametrize(
    "output_format",
    ["table", "json"],
)
def test_list_scorers_long_names(runner, experiment, generic_scorer, output_format):
    long_name = "VeryLongScorerNameThatShouldNotBeTruncatedEvenIfItIsReallyReallyLong"
    generic_scorer.register(experiment_id=experiment, name=long_name)

    args = ["list", "--experiment-id", experiment]
    if output_format == "json":
        args.extend(["--output", "json"])

    result = runner.invoke(commands, args)
    assert result.exit_code == 0

    if output_format == "json":
        output_json = json.loads(result.output)
        assert output_json == {"scorers": [long_name]}
    else:
        # Full name should be present
        assert long_name in result.output
