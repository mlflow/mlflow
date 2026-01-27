"""Tests for the mlflow demo CLI command.

Includes both quick help/registration tests and functional tests that
invoke the actual CLI command with a mocked server.
"""

import socket
import sys
from unittest import mock

import click
import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli import cli
from mlflow.cli.demo import _check_server_connection, demo
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DEMO_PROMPT_PREFIX
from mlflow.demo.generators.traces import DEMO_VERSION_TAG
from mlflow.demo.registry import demo_registry
from mlflow.genai.datasets import search_datasets
from mlflow.genai.prompts import search_prompts


@pytest.fixture(autouse=True)
def disable_quiet_logging(monkeypatch):
    """Prevent CLI from modifying logging state during tests."""
    demo_module = sys.modules["mlflow.cli.demo"]
    monkeypatch.setattr(demo_module, "_set_quiet_logging", lambda: None)


def test_demo_command_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "--help"])

    assert result.exit_code == 0
    assert "Launch MLflow with pre-populated demo data" in result.output


def test_demo_command_help_shows_options():
    runner = CliRunner()
    result = runner.invoke(demo, ["--help"])

    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--no-browser" in result.output


def test_demo_command_port_option():
    runner = CliRunner()
    result = runner.invoke(demo, ["--help"])

    assert result.exit_code == 0
    assert "Port to run demo server on" in result.output


def test_cli_generates_all_registered_features():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server"):
        result = runner.invoke(demo, ["--no-browser"], input="n\n")

    assert result.exit_code == 0
    assert "Generated:" in result.output

    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    registered_names = set(demo_registry.list_generators())
    for name in registered_names:
        assert name in result.output


def test_cli_creates_experiment():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server"):
        result = runner.invoke(demo, ["--no-browser"], input="n\n")

    assert result.exit_code == 0

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_cli_creates_traces():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server"):
        result = runner.invoke(demo, ["--no-browser"], input="n\n")

    assert result.exit_code == 0

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    all_traces = client.search_traces(
        locations=[experiment.experiment_id],
        max_results=200,
    )

    # Filter for demo traces only (exclude evaluation traces created by evaluate())
    demo_traces = [t for t in all_traces if t.info.trace_metadata.get(DEMO_VERSION_TAG)]
    assert len(demo_traces) == 34


def test_cli_creates_evaluation_datasets():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server"):
        result = runner.invoke(demo, ["--no-browser"], input="n\n")

    assert result.exit_code == 0

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    datasets = search_datasets(
        experiment_ids=[experiment.experiment_id],
        filter_string="name LIKE 'demo-%'",
        max_results=10,
    )

    assert len(datasets) == 2


def test_cli_creates_prompts():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server"):
        result = runner.invoke(demo, ["--no-browser"], input="n\n")

    assert result.exit_code == 0

    prompts = search_prompts(
        filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
        max_results=100,
    )

    assert len(prompts) == 3


def test_cli_shows_server_url():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server"):
        result = runner.invoke(demo, ["--no-browser"], input="n\n")

    assert result.exit_code == 0
    assert "MLflow Tracking Server running at:" in result.output
    assert "View the demo at:" in result.output


def test_cli_respects_port_option():
    runner = CliRunner()

    with mock.patch("mlflow.server._run_server") as mock_server:
        result = runner.invoke(demo, ["--no-browser", "--port", "5555"], input="n\n")

    assert result.exit_code == 0
    assert "http://127.0.0.1:5555" in result.output
    mock_server.assert_called_once()
    assert mock_server.call_args.kwargs["port"] == 5555


def test_cli_port_in_use_error():
    runner = CliRunner()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        bound_port = s.getsockname()[1]

        result = runner.invoke(demo, ["--port", str(bound_port)], input="n\n")

    assert result.exit_code != 0
    assert "already in use" in result.output


def test_cli_unreachable_server_error():
    runner = CliRunner()

    # Use a URL that won't have a server running
    result = runner.invoke(demo, ["--tracking-uri", "http://localhost:59999"])

    assert result.exit_code != 0
    assert "Cannot connect to MLflow server" in result.output
    assert "Please verify" in result.output


def test_check_server_connection_fails_for_bad_url():
    with pytest.raises(click.ClickException, match="Cannot connect to MLflow server"):
        _check_server_connection("http://localhost:59999", max_retries=1, timeout=1)
