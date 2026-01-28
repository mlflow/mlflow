from unittest import mock

import mlflow
from mlflow.genai.utils import display_utils
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_WORKSPACE_URL


def test_display_outputs_jupyter(monkeypatch):
    mock_store = mock.MagicMock(spec=RestStore)
    mock_store.get_run = MlflowClient().get_run
    mock_store.get_host_creds = lambda: mock.MagicMock(host="https://mlflow.example.com/")

    with (
        mock.patch("IPython.display.display") as mock_display,
        mock.patch.object(display_utils, "_get_store", return_value=mock_store),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
            return_value=True,
        ),
        mlflow.start_run() as run,
    ):
        display_utils.display_evaluation_output(run.info.run_id)

    exp_id = run.info.experiment_id
    expected_url = f"https://mlflow.example.com/#/experiments/{exp_id}/evaluation-runs?selectedRunUuid={run.info.run_id}"
    html_content = mock_display.call_args[0][0].data
    assert expected_url in html_content


def test_display_outputs_non_ipython(capsys):
    mock_store = mock.MagicMock(spec=RestStore)
    mock_store.get_run = mlflow.tracking.MlflowClient().get_run
    mock_store.get_host_creds = lambda: mock.MagicMock(host="https://mlflow.example.com/")

    with (
        mock.patch.object(display_utils, "_get_store", return_value=mock_store),
        mock.patch(
            "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
            return_value=False,
        ),
        mlflow.start_run() as run,
    ):
        display_utils.display_evaluation_output(run.info.run_id)

    captured = capsys.readouterr().out
    exp_id = run.info.experiment_id
    expected_url = f"https://mlflow.example.com/#/experiments/{exp_id}/evaluation-runs?selectedRunUuid={run.info.run_id}"
    assert expected_url in captured


def test_display_outputs_databricks(monkeypatch):
    host = "https://workspace.databricks.com"
    client = mlflow.tracking.MlflowClient()

    mock_store = mock.MagicMock(spec=RestStore)
    mock_store.get_run = client.get_run
    mock_store.get_host_creds = lambda: mock.MagicMock(host=host)

    with mlflow.start_run() as run:
        client.set_tag(run.info.run_id, MLFLOW_DATABRICKS_WORKSPACE_URL, host)

        with (
            mock.patch("IPython.display.display") as mock_display,
            mock.patch.object(display_utils, "_get_store", return_value=mock_store),
            mock.patch(
                "mlflow.tracking.context.jupyter_notebook_context._is_in_jupyter_notebook",
                return_value=True,
            ),
            mock.patch.object(display_utils, "is_databricks_uri", return_value=True),
        ):
            display_utils.display_evaluation_output(run.info.run_id)

    exp_id = run.info.experiment_id
    expected_url = (
        f"{host}/ml/experiments/{exp_id}/evaluation-runs?selectedRunUuid={run.info.run_id}"
    )
    html_content = mock_display.call_args[0][0].data
    assert expected_url in html_content


def test_display_summary_with_local_store(capsys):
    with mlflow.start_run() as run:
        display_utils.display_evaluation_output(run.info.run_id)

    captured = capsys.readouterr().out
    assert run.info.run_id in captured
    assert "Traces" in captured
