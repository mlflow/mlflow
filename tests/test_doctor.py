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
        assert "Active run ID: {}".format(run.info.run_id) in captured.out


def test_doctor_databricks_runtime(capsys):
    with mock.patch(
        "mlflow._doctor.get_databricks_runtime",
        return_value="12.0-cpu-ml-scala2.12",
    ) as mock_runtime:
        mlflow.doctor()
        mock_runtime.assert_called_once()
        captured = capsys.readouterr()
        assert "Databricks runtime version: 12.0-cpu-ml-scala2.12" in captured.out
