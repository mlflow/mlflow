import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mlflow.cli.datasets import commands


@pytest.fixture
def runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture
def mock_dataset():
    ds = MagicMock()
    ds.dataset_id = "ds-12345"
    ds.name = "qa_dataset"
    ds.digest = "abc123"
    ds.created_time = 1705312200000  # 2024-01-15 10:30:00 UTC
    ds.last_update_time = 1705412400000  # 2024-01-16 14:20:00 UTC
    ds.created_by = "user@example.com"
    ds.last_updated_by = "editor@example.com"
    ds.tags = {"env": "production"}
    return ds


@pytest.fixture
def mock_datasets_list(mock_dataset):
    datasets = MagicMock()
    datasets.__iter__ = lambda self: iter([mock_dataset])
    datasets.token = None
    return datasets


def test_commands_group_exists():
    assert commands.name == "datasets"
    assert commands.help is not None


def test_list_command_params():
    list_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "list"), None)
    assert list_cmd is not None
    param_names = {p.name for p in list_cmd.params}
    expected_params = {
        "experiment_id",
        "filter_string",
        "max_results",
        "order_by",
        "page_token",
        "output",
    }
    assert param_names == expected_params


def test_list_datasets_table_output(runner: CliRunner, mock_dataset: Any, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123"])

        assert result.exit_code == 0
        mock_client.search_datasets.assert_called_once_with(
            experiment_ids=["exp-123"],
            filter_string=None,
            max_results=50,
            order_by=None,
            page_token=None,
        )

        assert "ds-12345" in result.output
        assert "qa_dataset" in result.output
        assert "user@example.com" in result.output


def test_list_datasets_json_output(runner: CliRunner, mock_dataset: Any, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123", "--output", "json"])

        assert result.exit_code == 0

        output_json = json.loads(result.output)
        assert "datasets" in output_json
        assert "next_page_token" in output_json
        assert len(output_json["datasets"]) == 1

        ds = output_json["datasets"][0]
        assert ds["dataset_id"] == "ds-12345"
        assert ds["name"] == "qa_dataset"
        assert ds["digest"] == "abc123"
        assert ds["created_time"] == 1705312200000
        assert ds["last_update_time"] == 1705412400000
        assert ds["created_by"] == "user@example.com"
        assert ds["last_updated_by"] == "editor@example.com"
        assert ds["tags"] == {"env": "production"}


def test_list_datasets_empty_results(runner: CliRunner):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        empty_datasets = MagicMock()
        empty_datasets.__iter__ = lambda self: iter([])
        empty_datasets.token = None
        mock_client.search_datasets.return_value = empty_datasets
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123"])

        assert result.exit_code == 0
        mock_client.search_datasets.assert_called_once()


def test_list_datasets_json_empty_results(runner: CliRunner):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        empty_datasets = MagicMock()
        empty_datasets.__iter__ = lambda self: iter([])
        empty_datasets.token = None
        mock_client.search_datasets.return_value = empty_datasets
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json == {"datasets": [], "next_page_token": None}


def test_list_datasets_with_experiment_id_env_var(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list"], env={"MLFLOW_EXPERIMENT_ID": "exp-from-env"})

        assert result.exit_code == 0
        mock_client.search_datasets.assert_called_once()
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["experiment_ids"] == ["exp-from-env"]


def test_list_datasets_missing_experiment_id(runner: CliRunner):
    result = runner.invoke(commands, ["list"])

    assert result.exit_code != 0
    assert "experiment-id" in result.output.lower() or "experiment_id" in result.output.lower()


def test_list_datasets_invalid_output_format(runner: CliRunner):
    result = runner.invoke(commands, ["list", "--experiment-id", "exp-123", "--output", "invalid"])

    assert result.exit_code != 0
    assert "invalid" in result.output.lower() or "choice" in result.output.lower()


def test_list_datasets_with_filter_string(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            commands, ["list", "--experiment-id", "exp-123", "--filter-string", "name LIKE 'qa_%'"]
        )

        assert result.exit_code == 0
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["filter_string"] == "name LIKE 'qa_%'"


def test_list_datasets_with_max_results(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            commands, ["list", "--experiment-id", "exp-123", "--max-results", "100"]
        )

        assert result.exit_code == 0
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["max_results"] == 100


def test_list_datasets_with_order_by(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            commands, ["list", "--experiment-id", "exp-123", "--order-by", "last_update_time DESC"]
        )

        assert result.exit_code == 0
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["order_by"] == ["last_update_time DESC"]


def test_list_datasets_with_multiple_order_by(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            commands,
            ["list", "--experiment-id", "exp-123", "--order-by", "name ASC, created_time DESC"],
        )

        assert result.exit_code == 0
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["order_by"] == ["name ASC", "created_time DESC"]


def test_list_datasets_with_page_token(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            commands, ["list", "--experiment-id", "exp-123", "--page-token", "next-page-token"]
        )

        assert result.exit_code == 0
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["page_token"] == "next-page-token"


def test_list_datasets_displays_pagination_token(runner: CliRunner, mock_dataset: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        datasets_with_token = MagicMock()
        datasets_with_token.__iter__ = lambda self: iter([mock_dataset])
        datasets_with_token.token = "next-page-abc123"
        mock_client.search_datasets.return_value = datasets_with_token
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123"])

        assert result.exit_code == 0
        assert "Next page token: next-page-abc123" in result.output


def test_list_datasets_json_includes_pagination_token(runner: CliRunner, mock_dataset: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        datasets_with_token = MagicMock()
        datasets_with_token.__iter__ = lambda self: iter([mock_dataset])
        datasets_with_token.token = "next-page-abc123"
        mock_client.search_datasets.return_value = datasets_with_token
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json["next_page_token"] == "next-page-abc123"


def test_list_datasets_with_null_optional_fields(runner: CliRunner):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        ds = MagicMock()
        ds.dataset_id = "ds-99999"
        ds.name = "sparse_dataset"
        ds.digest = "xyz789"
        ds.created_time = None
        ds.last_update_time = None
        ds.created_by = None
        ds.last_updated_by = None
        ds.tags = None

        datasets = MagicMock()
        datasets.__iter__ = lambda self: iter([ds])
        datasets.token = None
        mock_client.search_datasets.return_value = datasets
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123"])

        assert result.exit_code == 0
        assert "ds-99999" in result.output
        assert "sparse_dataset" in result.output


def test_list_datasets_json_with_null_optional_fields(runner: CliRunner):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        ds = MagicMock()
        ds.dataset_id = "ds-99999"
        ds.name = "sparse_dataset"
        ds.digest = "xyz789"
        ds.created_time = None
        ds.last_update_time = None
        ds.created_by = None
        ds.last_updated_by = None
        ds.tags = None

        datasets = MagicMock()
        datasets.__iter__ = lambda self: iter([ds])
        datasets.token = None
        mock_client.search_datasets.return_value = datasets
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123", "--output", "json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        ds_output = output_json["datasets"][0]
        assert ds_output["dataset_id"] == "ds-99999"
        assert ds_output["name"] == "sparse_dataset"
        assert ds_output["created_time"] is None
        assert ds_output["last_update_time"] is None
        assert ds_output["created_by"] is None
        assert ds_output["last_updated_by"] is None
        assert ds_output["tags"] is None


def test_list_datasets_multiple_datasets(runner: CliRunner):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()

        ds1 = MagicMock()
        ds1.dataset_id = "ds-001"
        ds1.name = "dataset_one"
        ds1.digest = "hash1"
        ds1.created_time = 1705312200000
        ds1.last_update_time = 1705312200000
        ds1.created_by = "alice@example.com"
        ds1.last_updated_by = "alice@example.com"
        ds1.tags = {}

        ds2 = MagicMock()
        ds2.dataset_id = "ds-002"
        ds2.name = "dataset_two"
        ds2.digest = "hash2"
        ds2.created_time = 1705398600000
        ds2.last_update_time = 1705398600000
        ds2.created_by = "bob@example.com"
        ds2.last_updated_by = "charlie@example.com"
        ds2.tags = {"type": "test"}

        datasets = MagicMock()
        datasets.__iter__ = lambda self: iter([ds1, ds2])
        datasets.token = None
        mock_client.search_datasets.return_value = datasets
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "--experiment-id", "exp-123"])

        assert result.exit_code == 0
        assert "ds-001" in result.output
        assert "dataset_one" in result.output
        assert "alice@example.com" in result.output
        assert "ds-002" in result.output
        assert "dataset_two" in result.output
        assert "bob@example.com" in result.output


def test_list_datasets_short_option_x(runner: CliRunner, mock_datasets_list: Any):
    with patch("mlflow.cli.datasets.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.search_datasets.return_value = mock_datasets_list
        mock_client_class.return_value = mock_client

        result = runner.invoke(commands, ["list", "-x", "exp-short"])

        assert result.exit_code == 0
        call_args = mock_client.search_datasets.call_args
        assert call_args[1]["experiment_ids"] == ["exp-short"]
