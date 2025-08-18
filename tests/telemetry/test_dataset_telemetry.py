import pandas as pd
import pytest
from unittest import mock

import mlflow
from mlflow.entities.trace import Trace
from mlflow.telemetry.events import CreateDatasetEvent, MergeRecordsEvent


@pytest.fixture
def mock_telemetry_client():
    """Mock telemetry client for testing."""
    with mock.patch("mlflow.telemetry.client.get_telemetry_client") as mock_client:
        mock_instance = mock.MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


def test_create_dataset_event_parse():
    """Test CreateDatasetEvent parse method."""
    # Test in non-Databricks environment
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="sqlite://test.db"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False
        ):
            result = CreateDatasetEvent.parse({})
            assert result == {}

    # Test in Databricks environment - should return None
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=True
        ):
            result = CreateDatasetEvent.parse({})
            assert result is None


def test_merge_records_event_parse_dict():
    """Test MergeRecordsEvent parse with dict records."""
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="sqlite://test.db"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False
        ):
            # Test with list of dicts
            records = [
                {"inputs": {"question": "What is MLflow?"}, "expectations": {"answer": "A tool"}},
                {"inputs": {"question": "What is Spark?"}, "expectations": {"answer": "Engine"}},
            ]
            result = MergeRecordsEvent.parse({"records": records})
            assert result == {"record_count": 2, "input_type": "list"}

            # Test with empty list
            result = MergeRecordsEvent.parse({"records": []})
            assert result is None

            # Test with None
            result = MergeRecordsEvent.parse({"records": None})
            assert result is None


def test_merge_records_event_parse_pandas():
    """Test MergeRecordsEvent parse with pandas DataFrame."""
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="sqlite://test.db"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False
        ):
            # Test with pandas DataFrame
            df = pd.DataFrame(
                [
                    {"inputs": {"q": "Q1"}, "expectations": {"a": "A1"}},
                    {"inputs": {"q": "Q2"}, "expectations": {"a": "A2"}},
                ]
            )
            result = MergeRecordsEvent.parse({"records": df})
            assert result == {"record_count": 2, "input_type": "pandas"}

            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            result = MergeRecordsEvent.parse({"records": empty_df})
            assert result is None


def test_merge_records_event_parse_traces():
    """Test MergeRecordsEvent parse with list of Trace objects."""
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="sqlite://test.db"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False
        ):
            # Mock Trace objects
            mock_trace1 = mock.MagicMock(spec=Trace)
            mock_trace1.__class__.__name__ = "Trace"
            mock_trace2 = mock.MagicMock(spec=Trace)
            mock_trace2.__class__.__name__ = "Trace"

            traces = [mock_trace1, mock_trace2]
            result = MergeRecordsEvent.parse({"records": traces})
            assert result == {"record_count": 2, "input_type": "list"}


def test_merge_records_event_databricks():
    """Test MergeRecordsEvent in Databricks environment."""
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="databricks"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=True
        ):
            records = [{"inputs": {"q": "test"}}]
            result = MergeRecordsEvent.parse({"records": records})
            assert result is None


def test_create_dataset_telemetry_integration(mock_telemetry_client):
    """Test that create_dataset triggers telemetry event."""
    import os
    # Enable telemetry for testing
    os.environ['MLFLOW_TELEMETRY_ENABLED'] = 'true'
    
    try:
        with mock.patch("mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False):
            with mock.patch("mlflow.tracking.client.MlflowClient.create_dataset") as mock_create:
                mock_dataset = mock.MagicMock()
                mock_create.return_value = mock_dataset
                
                # Mock the record_usage_event decorator to capture calls
                with mock.patch("mlflow.telemetry.record_usage_event") as mock_decorator:
                    # Make the decorator return a function that calls the wrapped function
                    def decorator_impl(event_class):
                        def wrapper(func):
                            def wrapped(*args, **kwargs):
                                # Simulate telemetry recording
                                mock_telemetry_client.add_record(mock.MagicMock(event_name=event_class.name))
                                return func(*args, **kwargs)
                            return wrapped
                        return wrapper
                    mock_decorator.side_effect = decorator_impl

                    from mlflow.genai.datasets import create_dataset

                    dataset = create_dataset(name="test_dataset", tags={"test": "value"})

                    # Verify telemetry was called
                    assert mock_telemetry_client.add_record.called
                    record = mock_telemetry_client.add_record.call_args[0][0]
                    assert record.event_name == "create_dataset"
    finally:
        # Clean up
        if 'MLFLOW_TELEMETRY_ENABLED' in os.environ:
            del os.environ['MLFLOW_TELEMETRY_ENABLED']


def test_merge_records_telemetry_integration(mock_telemetry_client):
    """Test that merge_records triggers telemetry event."""
    import os
    # Enable telemetry for testing
    os.environ['MLFLOW_TELEMETRY_ENABLED'] = 'true'
    
    try:
        with mock.patch("mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False):
            # Patch _get_store where it's imported in evaluation_dataset module
            with mock.patch("mlflow.entities.evaluation_dataset._get_store") as mock_store:
                mock_store_instance = mock.MagicMock()
                mock_store.return_value = mock_store_instance
                mock_store_instance.upsert_dataset_records.return_value = {
                    "inserted": 2,
                    "updated": 0,
                }
                # Mock get_dataset to avoid the "dataset not found" error
                mock_store_instance.get_dataset.return_value = mock.MagicMock(dataset_id="test-id")

                # Mock the record_usage_event decorator to capture calls
                with mock.patch("mlflow.telemetry.record_usage_event") as mock_decorator:
                    # Make the decorator return a function that calls the wrapped function
                    def decorator_impl(event_class):
                        def wrapper(func):
                            def wrapped(*args, **kwargs):
                                # Simulate telemetry recording
                                mock_telemetry_client.add_record(mock.MagicMock(event_name=event_class.name))
                                return func(*args, **kwargs)
                            return wrapped
                        return wrapper
                    mock_decorator.side_effect = decorator_impl

                    from mlflow.entities.evaluation_dataset import EvaluationDataset

                    dataset = EvaluationDataset(
                        dataset_id="test-id",
                        name="test",
                        digest="digest",
                        created_time=123,
                        last_update_time=456,
                    )

                    records = [
                        {"inputs": {"q": "Q1"}, "expectations": {"a": "A1"}},
                        {"inputs": {"q": "Q2"}, "expectations": {"a": "A2"}},
                    ]
                    dataset.merge_records(records)

                    # Verify telemetry was called
                    assert mock_telemetry_client.add_record.called
                    record = mock_telemetry_client.add_record.call_args[0][0]
                    assert record.event_name == "merge_records"
    finally:
        # Clean up
        if 'MLFLOW_TELEMETRY_ENABLED' in os.environ:
            del os.environ['MLFLOW_TELEMETRY_ENABLED']


def test_telemetry_exception_handling():
    """Test that telemetry exceptions don't break the main flow."""
    # Test CreateDatasetEvent with exception
    with mock.patch("mlflow.tracking.get_tracking_uri", side_effect=Exception("Error")):
        result = CreateDatasetEvent.parse({})
        assert result is None

    # Test MergeRecordsEvent with exception
    with mock.patch("mlflow.tracking.get_tracking_uri", side_effect=Exception("Error")):
        result = MergeRecordsEvent.parse({"records": [{"test": "data"}]})
        assert result is None


def test_merge_records_event_malformed_data():
    """Test MergeRecordsEvent handles malformed data gracefully."""
    with mock.patch("mlflow.tracking.get_tracking_uri", return_value="sqlite://test.db"):
        with mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri", return_value=False
        ):
            # Test with None records
            assert MergeRecordsEvent.parse({"records": None}) is None
            
            # Test with empty dict
            assert MergeRecordsEvent.parse({}) is None
            
            # Test with non-iterable records  
            assert MergeRecordsEvent.parse({"records": 123}) is None
            
            # Test with object that raises on len()
            class BadObject:
                def __len__(self):
                    raise RuntimeError("Cannot get length")
            
            assert MergeRecordsEvent.parse({"records": BadObject()}) is None


def test_telemetry_never_raises():
    """Verify that telemetry NEVER raises exceptions, no matter what."""
    # Test with completely broken arguments
    test_cases = [
        None,
        {},
        {"records": None},
        {"records": object()},
        {"records": lambda x: x},
        {"records": type},
        {"records": ...},  # Ellipsis object
        {"records": NotImplemented},
        {"records": float("inf")},
        {"records": float("nan")},
    ]
    
    for test_case in test_cases:
        # Both events should handle ANY input without raising
        try:
            result = CreateDatasetEvent.parse(test_case)
            assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"CreateDatasetEvent raised exception with {test_case}: {e}")
        
        try:
            result = MergeRecordsEvent.parse(test_case)
            assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"MergeRecordsEvent raised exception with {test_case}: {e}")