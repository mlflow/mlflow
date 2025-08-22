import tempfile
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.entities.evaluation_dataset import EvaluationDataset
from mlflow.genai.datasets import create_dataset
from mlflow.telemetry.events import CreateDatasetEvent, MergeRecordsEvent


@pytest.fixture
def enable_telemetry(monkeypatch):
    monkeypatch.setenv("MLFLOW_TELEMETRY_ENABLED", "true")


@pytest.fixture
def mock_non_databricks():
    with mock.patch(
        "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False
    ):
        yield


@pytest.fixture
def mock_telemetry_client():
    mock_client = mock.MagicMock()
    with (
        mock.patch("mlflow.telemetry.track.get_telemetry_client", return_value=mock_client),
        mock.patch("mlflow.telemetry.track.is_telemetry_disabled", return_value=False),
    ):
        yield mock_client


@pytest.fixture
def mock_dataset_store():
    with mock.patch("mlflow.entities.evaluation_dataset._get_store") as mock_store:
        mock_store_instance = mock.MagicMock()
        mock_store.return_value = mock_store_instance
        mock_store_instance.get_dataset.return_value = mock.MagicMock(dataset_id="test-id")
        yield mock_store_instance


@pytest.fixture
def evaluation_dataset():
    return EvaluationDataset(
        dataset_id="test-id",
        name="test",
        digest="digest",
        created_time=123,
        last_update_time=456,
    )


@pytest.fixture
def sqlite_tracking_uri():
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow.set_tracking_uri(f"sqlite:///{temp_dir}/test.db")
        yield


def test_create_dataset_telemetry_integration(
    enable_telemetry, mock_non_databricks, sqlite_tracking_uri, mock_telemetry_client
):
    create_dataset(name="test_dataset", tags={"test": "value"})

    assert mock_telemetry_client.add_record.called
    record = mock_telemetry_client.add_record.call_args[0][0]
    assert record.event_name == "create_dataset"


def test_merge_records_telemetry_integration(
    enable_telemetry,
    mock_non_databricks,
    mock_telemetry_client,
    mock_dataset_store,
    evaluation_dataset,
):
    mock_dataset_store.upsert_dataset_records.return_value = {"inserted": 2, "updated": 0}

    records = [
        {"inputs": {"q": "Q1"}, "expectations": {"a": "A1"}},
        {"inputs": {"q": "Q2"}, "expectations": {"a": "A2"}},
    ]
    evaluation_dataset.merge_records(records)

    assert mock_telemetry_client.add_record.called
    record = mock_telemetry_client.add_record.call_args[0][0]
    assert record.event_name == "merge_records"
    assert record.params == {"record_count": 2, "input_type": "dict"}


def test_merge_records_with_dataframe(
    enable_telemetry,
    mock_non_databricks,
    mock_telemetry_client,
    mock_dataset_store,
    evaluation_dataset,
):
    mock_dataset_store.upsert_dataset_records.return_value = {"inserted": 3, "updated": 0}

    df = pd.DataFrame(
        [
            {"inputs": {"q": "Q3"}, "expectations": {"a": "A3"}},
            {"inputs": {"q": "Q4"}, "expectations": {"a": "A4"}},
            {"inputs": {"q": "Q5"}, "expectations": {"a": "A5"}},
        ]
    )
    evaluation_dataset.merge_records(df)

    assert mock_telemetry_client.add_record.called
    record = mock_telemetry_client.add_record.call_args[0][0]
    assert record.event_name == "merge_records"
    assert record.params == {"record_count": 3, "input_type": "pandas"}


def test_merge_records_with_traces(
    enable_telemetry,
    mock_non_databricks,
    mock_telemetry_client,
    mock_dataset_store,
    evaluation_dataset,
):
    mock_dataset_store.upsert_dataset_records.return_value = {"inserted": 2, "updated": 0}

    from mlflow.entities.trace import Trace

    trace1 = mock.MagicMock(spec=Trace)
    trace2 = mock.MagicMock(spec=Trace)
    traces = [trace1, trace2]

    processed_records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform"},
        },
        {
            "inputs": {"question": "What is Spark?"},
            "expectations": {"answer": "Spark is a data processing engine"},
        },
    ]

    with mock.patch.object(
        evaluation_dataset, "_process_trace_records", return_value=processed_records
    ):
        evaluation_dataset.merge_records(traces)

    assert mock_telemetry_client.add_record.called
    record = mock_telemetry_client.add_record.call_args[0][0]
    assert record.event_name == "merge_records"
    assert record.params == {"record_count": 2, "input_type": "trace"}


def test_telemetry_exception_handling():
    # CreateDatasetEvent has no parse method, returns None (default from base Event class)
    result = CreateDatasetEvent.parse({})
    assert result is None

    # MergeRecordsEvent returns None when records have no length
    result = MergeRecordsEvent.parse({"records": object()})
    assert result is None


@pytest.mark.parametrize(
    "test_input",
    [
        None,
        {},
        {"records": None},
        {"records": object()},
        {"records": lambda x: x},
        {"records": type},
        {"records": ...},
        {"records": NotImplemented},
        {"records": float("inf")},
        {"records": float("nan")},
    ],
)
def test_telemetry_never_raises(test_input):
    # CreateDatasetEvent should handle all inputs without raising
    result = CreateDatasetEvent.parse(test_input)
    assert result is None or isinstance(result, dict)

    # MergeRecordsEvent should handle all inputs without raising
    result = MergeRecordsEvent.parse(test_input)
    assert result is None or isinstance(result, dict)
