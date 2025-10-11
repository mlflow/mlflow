import json

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli.scorers import commands
from mlflow.genai.scorers import scorer


@pytest.fixture
def runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture
def experiment_with_scorers():
    """Create an experiment with registered scorers for testing."""
    experiment_id = mlflow.create_experiment(
        f"test_scorers_cli_{mlflow.utils.time.get_current_time_millis()}"
    )

    # Register a few test scorers
    @scorer
    def correctness_scorer(outputs) -> bool:
        return len(outputs) > 0

    @scorer
    def safety_scorer(outputs) -> bool:
        return len(outputs) > 0

    @scorer
    def relevance_scorer(outputs) -> bool:
        return len(outputs) > 0

    correctness_scorer.register(experiment_id=experiment_id, name="Correctness")
    safety_scorer.register(experiment_id=experiment_id, name="Safety")
    relevance_scorer.register(experiment_id=experiment_id, name="RelevanceToQuery")

    yield experiment_id

    # Cleanup
    mlflow.delete_experiment(experiment_id)


def test_commands_group_exists():
    assert commands.name == "scorers"
    assert commands.help is not None


def test_list_command_params():
    list_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "list"), None)
    assert list_cmd is not None
    param_names = {p.name for p in list_cmd.params}
    assert param_names == {"experiment_id", "output"}


def test_list_scorers_table_output(runner, experiment_with_scorers):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment_with_scorers])

    assert result.exit_code == 0
    assert "Scorer Name" in result.output
    assert "Correctness" in result.output
    assert "Safety" in result.output
    assert "RelevanceToQuery" in result.output


def test_list_scorers_json_output(runner, experiment_with_scorers):
    result = runner.invoke(
        commands, ["list", "--experiment-id", experiment_with_scorers, "--output", "json"]
    )

    assert result.exit_code == 0
    output_json = json.loads(result.output)
    assert set(output_json["scorers"]) == {"Correctness", "Safety", "RelevanceToQuery"}


def test_list_scorers_empty_experiment(runner):
    experiment_id = mlflow.create_experiment(
        f"test_empty_{mlflow.utils.time.get_current_time_millis()}"
    )

    try:
        result = runner.invoke(commands, ["list", "--experiment-id", experiment_id])
        assert result.exit_code == 0
        # Empty table produces minimal output
        assert result.output.strip() == ""
    finally:
        mlflow.delete_experiment(experiment_id)


def test_list_scorers_empty_experiment_json(runner):
    experiment_id = mlflow.create_experiment(
        f"test_empty_json_{mlflow.utils.time.get_current_time_millis()}"
    )

    try:
        result = runner.invoke(
            commands, ["list", "--experiment-id", experiment_id, "--output", "json"]
        )
        assert result.exit_code == 0
        output_json = json.loads(result.output)
        expected = {"scorers": []}
        assert output_json == expected
    finally:
        mlflow.delete_experiment(experiment_id)


def test_list_scorers_with_experiment_id_env_var(runner, experiment_with_scorers):
    result = runner.invoke(
        commands, ["list"], env={"MLFLOW_EXPERIMENT_ID": experiment_with_scorers}
    )

    assert result.exit_code == 0
    assert "Correctness" in result.output


def test_list_scorers_missing_experiment_id(runner):
    result = runner.invoke(commands, ["list"])

    assert result.exit_code != 0
    assert "experiment-id" in result.output.lower() or "experiment_id" in result.output.lower()


def test_list_scorers_invalid_output_format(runner, experiment_with_scorers):
    result = runner.invoke(
        commands, ["list", "--experiment-id", experiment_with_scorers, "--output", "invalid"]
    )

    assert result.exit_code != 0
    assert "invalid" in result.output.lower() or "choice" in result.output.lower()


def test_list_scorers_special_characters_in_names(runner):
    experiment_id = mlflow.create_experiment(
        f"test_special_{mlflow.utils.time.get_current_time_millis()}"
    )

    try:
        # Register scorers with special characters in names
        @scorer
        def test_scorer(outputs) -> bool:
            return True

        test_scorer.register(experiment_id=experiment_id, name="Scorer With Spaces")
        test_scorer.register(experiment_id=experiment_id, name="Scorer.With.Dots")
        test_scorer.register(experiment_id=experiment_id, name="Scorer-With-Dashes")
        test_scorer.register(experiment_id=experiment_id, name="Scorer_With_Underscores")

        result = runner.invoke(commands, ["list", "--experiment-id", experiment_id])

        assert result.exit_code == 0
        assert "Scorer With Spaces" in result.output
        assert "Scorer.With.Dots" in result.output
        assert "Scorer-With-Dashes" in result.output
        assert "Scorer_With_Underscores" in result.output
    finally:
        mlflow.delete_experiment(experiment_id)


def test_list_scorers_single_scorer(runner):
    experiment_id = mlflow.create_experiment(
        f"test_single_{mlflow.utils.time.get_current_time_millis()}"
    )

    try:

        @scorer
        def only_scorer(outputs) -> bool:
            return True

        only_scorer.register(experiment_id=experiment_id, name="OnlyScorer")

        result = runner.invoke(commands, ["list", "--experiment-id", experiment_id])

        assert result.exit_code == 0
        assert "OnlyScorer" in result.output
    finally:
        mlflow.delete_experiment(experiment_id)


def test_list_scorers_single_scorer_json(runner):
    experiment_id = mlflow.create_experiment(
        f"test_single_json_{mlflow.utils.time.get_current_time_millis()}"
    )

    try:

        @scorer
        def only_scorer(outputs) -> bool:
            return True

        only_scorer.register(experiment_id=experiment_id, name="OnlyScorer")

        result = runner.invoke(
            commands, ["list", "--experiment-id", experiment_id, "--output", "json"]
        )

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        expected = {"scorers": ["OnlyScorer"]}
        assert output_json == expected
    finally:
        mlflow.delete_experiment(experiment_id)


def test_list_scorers_long_names(runner):
    experiment_id = mlflow.create_experiment(
        f"test_long_{mlflow.utils.time.get_current_time_millis()}"
    )
    long_name = "VeryLongScorerNameThatShouldNotBeTruncatedEvenIfItIsReallyReallyLong"

    try:

        @scorer
        def long_scorer(outputs) -> bool:
            return True

        long_scorer.register(experiment_id=experiment_id, name=long_name)

        result = runner.invoke(commands, ["list", "--experiment-id", experiment_id])

        assert result.exit_code == 0
        # Full name should be present
        assert long_name in result.output
    finally:
        mlflow.delete_experiment(experiment_id)


def test_list_scorers_long_names_json(runner):
    experiment_id = mlflow.create_experiment(
        f"test_long_json_{mlflow.utils.time.get_current_time_millis()}"
    )
    long_name = "VeryLongScorerNameThatShouldNotBeTruncatedEvenIfItIsReallyReallyLong"

    try:

        @scorer
        def long_scorer(outputs) -> bool:
            return True

        long_scorer.register(experiment_id=experiment_id, name=long_name)

        result = runner.invoke(
            commands, ["list", "--experiment-id", experiment_id, "--output", "json"]
        )

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        expected = {"scorers": [long_name]}
        assert output_json == expected
    finally:
        mlflow.delete_experiment(experiment_id)
