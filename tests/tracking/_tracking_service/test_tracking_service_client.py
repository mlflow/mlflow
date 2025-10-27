from unittest import mock

import pytest

from mlflow.entities import Metric, Param, Run, RunInfo, RunTag
from mlflow.exceptions import MlflowException
from mlflow.tracking._tracking_service.client import TrackingServiceClient


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


def newTrackingServiceClient():
    return TrackingServiceClient("databricks://scope:key")


@pytest.mark.parametrize(
    ("artifact_uri", "databricks_uri", "uri_for_repo"),
    [
        ("dbfs:/path", "databricks://profile", "dbfs://profile@databricks/path"),
        ("dbfs:/path", "databricks://scope:key", "dbfs://scope:key@databricks/path"),
        ("runs:/path", "databricks://scope:key", "runs://scope:key@databricks/path"),
        ("models:/path", "databricks://scope:key", "models://scope:key@databricks/path"),
        # unaffected uri cases
        (
            "dbfs://profile@databricks/path",
            "databricks://scope:key",
            "dbfs://profile@databricks/path",
        ),
        (
            "dbfs://profile@databricks/path",
            "databricks://profile2",
            "dbfs://profile@databricks/path",
        ),
        ("s3:/path", "databricks://profile", "s3:/path"),
        ("ftp://user:pass@host/path", "databricks://profile", "ftp://user:pass@host/path"),
    ],
)
def test_get_artifact_repo(artifact_uri, databricks_uri, uri_for_repo):
    with (
        mock.patch(
            "mlflow.tracking._tracking_service.client.TrackingServiceClient.get_run",
            return_value=Run(
                RunInfo(
                    "uuid",
                    "expr_id",
                    "userid",
                    "status",
                    0,
                    10,
                    "active",
                    artifact_uri=artifact_uri,
                ),
                None,
            ),
        ),
        mock.patch(
            "mlflow.tracking._tracking_service.client.get_artifact_repository", return_value=None
        ) as get_repo_mock,
    ):
        client = TrackingServiceClient(databricks_uri)
        client._get_artifact_repo("some-run-id")
        get_repo_mock.assert_called_once_with(uri_for_repo)


def test_artifact_repo_is_cached_per_run_id(tmp_path):
    uri = "ftp://user:pass@host/path"
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.get_run",
        return_value=Run(
            RunInfo("uuid", "expr_id", "userid", "status", 0, 10, "active", artifact_uri=uri),
            None,
        ),
    ):
        tracking_uri = tmp_path.as_uri()
        artifact_repo = TrackingServiceClient(tracking_uri)._get_artifact_repo("some_run_id")
        another_artifact_repo = TrackingServiceClient(tracking_uri)._get_artifact_repo(
            "some_run_id"
        )
        assert artifact_repo is another_artifact_repo


@pytest.fixture
def tracking_client_log_batch(tmp_path):
    client = TrackingServiceClient(tmp_path.as_uri())
    exp_id = client.create_experiment("test_log_batch")
    run = client.create_run(exp_id)
    return client, run.info.run_id


def test_log_batch(tracking_client_log_batch):
    client, run_id = tracking_client_log_batch

    metrics = [
        Metric(key="metric1", value=1.0, timestamp=12345, step=0),
        Metric(key="metric2", value=2.0, timestamp=23456, step=1),
    ]

    params = [Param(key="param1", value="value1"), Param(key="param2", value="value2")]

    tags = [RunTag(key="tag1", value="value1"), RunTag(key="tag2", value="value2")]

    client.log_batch(run_id=run_id, metrics=metrics, params=params, tags=tags)
    run_data = client.get_run(run_id).data

    expected_tags = {tag.key: tag.value for tag in tags}
    expected_tags["mlflow.runName"] = run_data.tags["mlflow.runName"]

    assert run_data.metrics == {metric.key: metric.value for metric in metrics}
    assert run_data.params == {param.key: param.value for param in params}
    assert run_data.tags == expected_tags


def test_log_batch_with_empty_data(tracking_client_log_batch):
    client, run_id = tracking_client_log_batch

    client.log_batch(run_id=run_id, metrics=[], params=[], tags=[])
    run_data = client.get_run(run_id).data

    assert run_data.metrics == {}
    assert run_data.params == {}
    assert run_data.tags == {"mlflow.runName": run_data.tags["mlflow.runName"]}


def test_log_batch_with_numpy_array(tracking_client_log_batch):
    import numpy as np

    client, run_id = tracking_client_log_batch

    metrics = [Metric(key="metric1", value=np.array(1.0), timestamp=12345, step=0)]
    params = [Param(key="param1", value="value1")]
    tags = [RunTag(key="tag1", value="value1")]

    client.log_batch(run_id=run_id, metrics=metrics, params=params, tags=tags)
    run_data = client.get_run(run_id).data

    expected_tags = {tag.key: tag.value for tag in tags}
    expected_tags["mlflow.runName"] = run_data.tags["mlflow.runName"]

    assert run_data.metrics == {metric.key: metric.value for metric in metrics}
    assert run_data.params == {param.key: param.value for param in params}
    assert run_data.tags == expected_tags


def test_link_traces_to_run_validation():
    client = newTrackingServiceClient()

    with pytest.raises(MlflowException, match="run_id cannot be empty"):
        client.link_traces_to_run(["trace1", "trace2"], "")

    with pytest.raises(MlflowException, match="run_id cannot be empty"):
        client.link_traces_to_run(["trace1", "trace2"], None)

    trace_ids = [f"trace_{i}" for i in range(101)]
    with pytest.raises(MlflowException, match="Cannot link more than 100 traces to a run"):
        client.link_traces_to_run(trace_ids, "run_id")
