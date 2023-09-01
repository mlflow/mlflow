from unittest import mock

import mlflow


def test_doctor(capsys):
    mlflow.doctor()
    captured = capsys.readouterr()
    assert f"MLflow version: {mlflow.__version__}" in captured.out


def test_doctor_active_run(capsys):
    with mlflow.start_run() as run:
        mlflow.doctor()
        captured = capsys.readouterr()
        assert f"Active run ID: {run.info.run_id}" in captured.out


def test_doctor_databricks_runtime(capsys):
    mock_version = "12.0-cpu-ml-scala2.12"
    with mock.patch(
        "mlflow._doctor.get_databricks_runtime", return_value=mock_version
    ) as mock_runtime:
        mlflow.doctor()
        mock_runtime.assert_called_once()
        captured = capsys.readouterr()
        assert f"Databricks runtime version: {mock_version}" in captured.out
