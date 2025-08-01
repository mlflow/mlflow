import pytest
from mlflow.entities.evaluation_dataset import EvaluationDataset
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED, ErrorCode
from mlflow.store.tracking.file_store import FileStore


@pytest.fixture
def file_store(tmp_path):
    return FileStore(str(tmp_path))


def assert_feature_disabled_error(exc_info, method_name):
    assert exc_info.value.error_code == ErrorCode.Name(FEATURE_DISABLED)
    assert f"{method_name} is not supported with FileStore" in str(exc_info.value)


def test_create_evaluation_dataset_not_supported(file_store):
    dataset = EvaluationDataset(name="test_dataset")

    with pytest.raises(MlflowException) as exc_info:
        file_store.create_evaluation_dataset(dataset, experiment_ids=["0"])

    assert_feature_disabled_error(exc_info, "create_evaluation_dataset")


def test_get_evaluation_dataset_not_supported(file_store):
    with pytest.raises(MlflowException) as exc_info:
        file_store.get_evaluation_dataset("dataset_123")

    assert_feature_disabled_error(exc_info, "get_evaluation_dataset")


def test_delete_evaluation_dataset_not_supported(file_store):
    with pytest.raises(MlflowException) as exc_info:
        file_store.delete_evaluation_dataset("dataset_123")

    assert_feature_disabled_error(exc_info, "delete_evaluation_dataset")


def test_search_evaluation_datasets_not_supported(file_store):
    with pytest.raises(MlflowException) as exc_info:
        file_store.search_evaluation_datasets(experiment_ids=["0"])

    assert_feature_disabled_error(exc_info, "search_evaluation_datasets")


def test_upsert_evaluation_dataset_records_not_supported(file_store):
    records = [{"inputs": {"question": "test"}}]

    with pytest.raises(MlflowException) as exc_info:
        file_store.upsert_evaluation_dataset_records("dataset_123", records)

    assert_feature_disabled_error(exc_info, "upsert_evaluation_dataset_records")
