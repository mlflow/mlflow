import json
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli.scorers import commands
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig


@pytest.fixture
def runner():
    """Provide a CLI runner for testing."""
    return CliRunner(catch_exceptions=False)


def _create_mock_scorer(name, sample_rate=0.1, filter_string=None):
    """Helper to create a mock Scorer object."""
    scorer = Scorer(name=name)
    scorer._sampling_config = ScorerSamplingConfig(
        sample_rate=sample_rate, filter_string=filter_string
    )
    scorer._registered_backend = "test_backend"
    return scorer


def test_commands_group_exists():
    """Verify that the scorers command group exists."""
    assert commands.name == "scorers"
    assert commands.help is not None


def test_list_command_params():
    """Verify that the list command has the expected parameters."""
    list_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "list"), None)
    assert list_cmd is not None
    param_names = [p.name for p in list_cmd.params]
    assert "experiment_id" in param_names
    assert "output" in param_names


def test_list_scorers_table_output(runner):
    """Verify that the command correctly displays scorer names in table format."""
    mock_scorers = [
        _create_mock_scorer("Correctness"),
        _create_mock_scorer("Safety"),
        _create_mock_scorer("RelevanceToQuery"),
    ]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123"])

        assert result.exit_code == 0
        assert "Scorer Name" in result.output
        assert "Correctness" in result.output
        assert "Safety" in result.output
        assert "RelevanceToQuery" in result.output


def test_list_scorers_json_output(runner):
    """Verify that the command correctly outputs scorer names as valid JSON array."""
    mock_scorers = [
        _create_mock_scorer("Correctness"),
        _create_mock_scorer("Safety"),
        _create_mock_scorer("RelevanceToQuery"),
    ]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "scorers" in output_json
        assert output_json["scorers"] == ["Correctness", "Safety", "RelevanceToQuery"]


def test_list_scorers_empty_experiment(runner):
    """Verify graceful handling when an experiment has no registered scorers."""
    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = []
        result = runner.invoke(commands, ["list", "--experiment-id", "123"])

        assert result.exit_code == 0
        # Empty table produces minimal output
        assert result.output.strip() == ""


def test_list_scorers_empty_experiment_json(runner):
    """Verify JSON output for empty experiment."""
    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = []
        result = runner.invoke(commands, ["list", "--experiment-id", "123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json == {"scorers": []}


def test_list_scorers_with_experiment_id_flag(runner):
    """Verify that experiment ID can be specified via --experiment-id flag."""
    mock_scorers = [_create_mock_scorer("Correctness")]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "456"])

        assert result.exit_code == 0
        mock_list.assert_called_once_with(experiment_id="456")


def test_list_scorers_with_experiment_id_short_flag(runner):
    """Verify that experiment ID can be specified via -x short flag."""
    mock_scorers = [_create_mock_scorer("Safety")]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "-x", "789"])

        assert result.exit_code == 0
        mock_list.assert_called_once_with(experiment_id="789")


def test_list_scorers_with_experiment_id_env_var(runner):
    """Verify that experiment ID can be read from MLFLOW_EXPERIMENT_ID environment variable."""
    mock_scorers = [_create_mock_scorer("Correctness")]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list"], env={"MLFLOW_EXPERIMENT_ID": "999"})

        assert result.exit_code == 0
        mock_list.assert_called_once_with(experiment_id="999")


def test_list_scorers_multiple_scorers(runner):
    """Verify correct listing when multiple scorers are registered."""
    mock_scorers = [
        _create_mock_scorer("Scorer1"),
        _create_mock_scorer("Scorer2"),
        _create_mock_scorer("Scorer3"),
        _create_mock_scorer("Scorer4"),
        _create_mock_scorer("Scorer5"),
    ]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123"])

        assert result.exit_code == 0
        for scorer in mock_scorers:
            assert scorer.name in result.output


def test_list_scorers_missing_experiment_id(runner):
    """Verify appropriate error when experiment ID is not provided."""
    result = runner.invoke(commands, ["list"])

    assert result.exit_code != 0
    assert "experiment-id" in result.output.lower() or "experiment_id" in result.output.lower()


def test_list_scorers_invalid_output_format(runner):
    """Verify that invalid --output values are rejected."""
    result = runner.invoke(commands, ["list", "--experiment-id", "123", "--output", "invalid"])

    assert result.exit_code != 0
    assert "invalid" in result.output.lower() or "choice" in result.output.lower()


def test_list_scorers_special_characters_in_names(runner):
    """Verify proper handling of scorer names with special characters."""
    mock_scorers = [
        _create_mock_scorer("Scorer With Spaces"),
        _create_mock_scorer("Scorer.With.Dots"),
        _create_mock_scorer("Scorer-With-Dashes"),
        _create_mock_scorer("Scorer_With_Underscores"),
    ]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123"])

        assert result.exit_code == 0
        for scorer in mock_scorers:
            assert scorer.name in result.output


def test_list_scorers_json_array_structure(runner):
    """Verify JSON output is a proper array of strings."""
    mock_scorers = [
        _create_mock_scorer("Scorer1"),
        _create_mock_scorer("Scorer2"),
    ]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)
        assert "scorers" in output_json
        assert isinstance(output_json["scorers"], list)
        assert all(isinstance(name, str) for name in output_json["scorers"])


def test_list_scorers_single_scorer(runner):
    """Verify correct display when only one scorer is registered."""
    mock_scorers = [_create_mock_scorer("OnlyScorer")]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123"])

        assert result.exit_code == 0
        assert "OnlyScorer" in result.output


def test_list_scorers_single_scorer_json(runner):
    """Verify JSON output for single scorer."""
    mock_scorers = [_create_mock_scorer("OnlyScorer")]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json == {"scorers": ["OnlyScorer"]}


def test_list_scorers_long_names(runner):
    """Verify that very long scorer names are displayed in full without truncation."""
    long_name = "VeryLongScorerNameThatShouldNotBeTruncatedEvenIfItIsReallyReallyLong"
    mock_scorers = [_create_mock_scorer(long_name)]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123"])

        assert result.exit_code == 0
        # Full name should be present
        assert long_name in result.output


def test_list_scorers_long_names_json(runner):
    """Verify that long names are fully preserved in JSON output."""
    long_name = "VeryLongScorerNameThatShouldNotBeTruncatedEvenIfItIsReallyReallyLong"
    mock_scorers = [_create_mock_scorer(long_name)]

    with mock.patch("mlflow.cli.scorers.list_scorers") as mock_list:
        mock_list.return_value = mock_scorers
        result = runner.invoke(commands, ["list", "--experiment-id", "123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json["scorers"][0] == long_name
