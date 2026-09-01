from unittest import mock

from mlflow.data.evaluation_dataset_source import EvaluationDatasetSource


def test_evaluation_dataset_source_round_trips_version():
    source = EvaluationDatasetSource("d-12345678901234567890123456789012", version=3)
    assert source.to_dict() == {
        "dataset_id": "d-12345678901234567890123456789012",
        "version": 3,
    }

    restored = EvaluationDatasetSource.from_dict(source.to_dict())
    assert restored.to_dict() == source.to_dict()


def test_evaluation_dataset_source_loads_resolved_version():
    source = EvaluationDatasetSource("d-12345678901234567890123456789012", version=2)
    store = mock.Mock()
    store.get_dataset.return_value = object()

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        source.load()

    store.get_dataset.assert_called_once_with(
        "d-12345678901234567890123456789012", version=2
    )
