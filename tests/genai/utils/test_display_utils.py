from unittest import mock

import pytest

import mlflow
from mlflow.genai.utils import display_utils
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_WORKSPACE_URL


@pytest.fixture
def run_id():
    """Create an actual MLflow run and return its ID"""
    with mlflow.start_run() as run:
        return run.info.run_id


def test_display_outputs_jupyter(run_id, monkeypatch):
    mock_store = mock.MagicMock(spec=RestStore)
    mock_store.get_run = MlflowClient().get_run
    mock_store.get_host_creds = lambda: mock.MagicMock(host="https://mlflow.example.com/")

    mock_display = mock.MagicMock()
    monkeypatch.setattr("IPython.display.display", mock_display)

    with (
        mock.patch.object(display_utils, "_get_store", return_value=mock_store),
        mock.patch.object(display_utils, "_is_jupyter", return_value=True),
    ):
        display_utils.display_evaluation_output(run_id)

    exp_id = MlflowClient().get_run(run_id).info.experiment_id
    expected_url = f"https://mlflow.example.com/#/experiments/{exp_id}/evaluation-runs?selectedRunUuid={run_id}"
    html_content = mock_display.call_args[0][0].data
    assert expected_url in html_content


def test_display_outputs_non_ipython(run_id, capsys):
    mock_store = mock.MagicMock(spec=RestStore)
    mock_store.get_run = mlflow.tracking.MlflowClient().get_run
    mock_store.get_host_creds = lambda: mock.MagicMock(host="https://mlflow.example.com/")

    with (
        mock.patch.object(display_utils, "_get_store", return_value=mock_store),
        mock.patch.object(display_utils, "_is_jupyter", return_value=False),
    ):
        display_utils.display_evaluation_output(run_id)

    captured = capsys.readouterr().out
    exp_id = MlflowClient().get_run(run_id).info.experiment_id
    expected_url = f"https://mlflow.example.com/#/experiments/{exp_id}/evaluation-runs?selectedRunUuid={run_id}"
    assert expected_url in captured


def test_display_outputs_databricks(run_id, monkeypatch):
    host = "https://workspace.databricks.com"
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, MLFLOW_DATABRICKS_WORKSPACE_URL, host)

    mock_store = mock.MagicMock(spec=RestStore)
    mock_store.get_run = client.get_run
    mock_store.get_host_creds = lambda: mock.MagicMock(host=host)

    mock_display = mock.MagicMock()
    monkeypatch.setattr("IPython.display.display", mock_display)

    with (
        mock.patch.object(display_utils, "_get_store", return_value=mock_store),
        mock.patch.object(display_utils, "_is_jupyter", return_value=True),
        mock.patch.object(display_utils, "is_databricks_uri", return_value=True),
    ):
        display_utils.display_evaluation_output(run_id)

    exp_id = MlflowClient().get_run(run_id).info.experiment_id
    expected_url = f"{host}/ml/experiments/{exp_id}/evaluation-runs?selectedRunUuid={run_id}"
    html_content = mock_display.call_args[0][0].data
    assert expected_url in html_content


def test_display_summary_with_local_store(run_id, capsys):
    display_utils.display_evaluation_output(run_id)

    captured = capsys.readouterr().out
    assert run_id in captured
    assert "Traces" in captured
