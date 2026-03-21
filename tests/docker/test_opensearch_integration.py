"""Integration tests for MLflow with OpenSearch backend.

These tests use Docker Compose to stand up a real OpenSearch cluster
and verify end-to-end functionality.

Run with:
    pytest tests/docker/test_opensearch_integration.py -v
"""

import os
from datetime import timedelta

import pytest
from testcontainers.compose import DockerCompose
from testcontainers.core.wait_strategies import HttpWaitStrategy

import mlflow


@pytest.mark.parametrize("compose_file", ["docker-compose.opensearch-test.yaml"])
def test_opensearch_backend_integration(compose_file):
    """End-to-end test with real OpenSearch cluster.

    Verifies:
    1. MLflow server starts with OpenSearch backend
    2. Experiments can be created and searched
    3. Runs with metrics/params/tags can be logged and queried
    4. Traces can be recorded and searched
    """
    compose = DockerCompose(
        context=os.path.dirname(os.path.abspath(__file__)),
        compose_file_name=[compose_file],
    )
    compose.waiting_for({
        "mlflow": HttpWaitStrategy(5000, "/health")
        .for_status_code(200)
        .with_startup_timeout(timedelta(minutes=5))
    })

    with compose:
        base_url = "http://localhost:5000"
        mlflow.set_tracking_uri(base_url)

        # --- Experiment Operations ---
        exp_id = mlflow.create_experiment("opensearch-integration-test")
        mlflow.set_experiment(experiment_id=exp_id)

        exp = mlflow.get_experiment(exp_id)
        assert exp.name == "opensearch-integration-test"

        # --- Run Operations ---
        with mlflow.start_run(run_name="test-run") as run:
            mlflow.log_param("learning_rate", "0.01")
            mlflow.log_param("batch_size", "32")
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("loss", 0.05)
            mlflow.set_tag("env", "integration-test")
            mlflow.set_tag("model_type", "transformer")

        # Verify run data was persisted
        run_data = mlflow.get_run(run.info.run_id)
        assert run_data.data.params["learning_rate"] == "0.01"
        assert run_data.data.metrics["accuracy"] == 0.95

        # Search runs by metric
        runs = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string="metrics.accuracy > 0.9",
        )
        assert len(runs) == 1

        # Search runs by tag
        runs = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string="tags.env = 'integration-test'",
        )
        assert len(runs) == 1

        # --- Trace Operations ---
        @mlflow.trace
        def predict(model_input: list[str]) -> list[str]:
            return [f"prediction for {x}" for x in model_input]

        result = predict(["input1", "input2"])
        assert len(result) == 2

        # Allow time for async trace export
        import time

        time.sleep(3)

        # Search traces
        traces = mlflow.search_traces(experiment_ids=[exp_id])
        assert len(traces) >= 1
