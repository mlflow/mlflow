import mlflow


def test_verify_alembic_runs(tracking_uri_mock):
    mlflow.set_tracking_uri(tracking_uri_mock)
    mlflow.set_experiment("exp_name_example")
