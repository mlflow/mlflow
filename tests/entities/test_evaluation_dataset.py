from unittest import mock

import pytest

from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.dataset_record_source import DatasetRecordSource
from mlflow.entities.evaluation_dataset import EvaluationDataset
from mlflow.exceptions import MlflowException
from mlflow.protos.evaluation_datasets_pb2 import EvaluationDataset as ProtoEvaluationDataset


def test_evaluation_dataset_creation():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        tags={"environment": "test", "version": "1.0"},
        schema='{"fields": ["input", "output"]}',
        profile='{"count": 100}',
        digest="abc123",
        created_by="user1",
        last_updated_by="user2",
        experiment_ids=["exp1", "exp2"],
    )

    assert dataset.dataset_id == "dataset123"
    assert dataset.name == "test_dataset"
    assert dataset.tags == {"environment": "test", "version": "1.0"}
    assert dataset.schema == '{"fields": ["input", "output"]}'
    assert dataset.profile == '{"count": 100}'
    assert dataset.digest == "abc123"
    assert dataset.created_by == "user1"
    assert dataset.last_updated_by == "user2"
    assert dataset.experiment_ids == ["exp1", "exp2"]
    assert dataset.created_time is not None
    assert dataset.last_update_time is not None


def test_evaluation_dataset_auto_timestamps():
    dataset = EvaluationDataset(name="test_dataset")

    assert dataset.created_time is not None
    assert dataset.last_update_time is not None
    assert dataset.created_time > 0
    assert dataset.last_update_time > 0


def test_evaluation_dataset_tags_handling():
    dataset_with_tags = EvaluationDataset(
        name="dataset_with_tags", tags={"key1": "value1", "nested": {"key2": "value2"}}
    )
    assert dataset_with_tags.tags == {"key1": "value1", "nested": {"key2": "value2"}}

    dataset_no_tags = EvaluationDataset(name="dataset_no_tags")
    assert dataset_no_tags.tags is None

    dataset_empty_tags = EvaluationDataset(name="dataset_empty_tags", tags={})
    assert dataset_empty_tags.tags == {}


def test_evaluation_dataset_to_from_proto():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        schema='{"fields": ["input", "output"]}',
        profile='{"count": 100}',
        digest="abc123",
        created_time=123456789,
        last_update_time=987654321,
        created_by="user1",
        last_updated_by="user2",
        experiment_ids=["exp1", "exp2"],
    )

    proto = dataset.to_proto()
    assert isinstance(proto, ProtoEvaluationDataset)
    assert proto.dataset_id == "dataset123"
    assert proto.name == "test_dataset"
    assert proto.schema == '{"fields": ["input", "output"]}'
    assert proto.profile == '{"count": 100}'
    assert proto.digest == "abc123"
    assert proto.created_time == 123456789
    assert proto.last_update_time == 987654321
    assert proto.created_by == "user1"
    assert proto.last_updated_by == "user2"
    assert list(proto.experiment_ids) == ["exp1", "exp2"]

    dataset2 = EvaluationDataset.from_proto(proto)
    assert dataset2.dataset_id == dataset.dataset_id
    assert dataset2.name == dataset.name
    assert dataset2.schema == dataset.schema
    assert dataset2.profile == dataset.profile
    assert dataset2.digest == dataset.digest
    assert dataset2.created_time == dataset.created_time
    assert dataset2.last_update_time == dataset.last_update_time
    assert dataset2.created_by == dataset.created_by
    assert dataset2.last_updated_by == dataset.last_updated_by
    assert dataset2.experiment_ids == dataset.experiment_ids


def test_evaluation_dataset_to_from_dict():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        tags={"env": "prod"},
        experiment_ids=["exp1", "exp2"],
    )

    dataset._records = [
        DatasetRecord(dataset_id="dataset123", inputs={"question": "What is MLflow?"})
    ]

    data = dataset.to_dict()
    assert data["dataset_id"] == "dataset123"
    assert data["name"] == "test_dataset"
    assert data["tags"] == {"env": "prod"}
    assert data["experiment_ids"] == ["exp1", "exp2"]
    assert len(data["records"]) == 1
    assert data["records"][0]["inputs"]["question"] == "What is MLflow?"

    dataset2 = EvaluationDataset.from_dict(data)
    assert dataset2.dataset_id == dataset.dataset_id
    assert dataset2.name == dataset.name
    assert dataset2.tags == dataset.tags
    assert dataset2.experiment_ids == dataset.experiment_ids
    assert len(dataset2._records) == 1
    assert dataset2._records[0].inputs["question"] == "What is MLflow?"


def test_evaluation_dataset_records_property():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    assert dataset.has_records() is False
    assert dataset._records is None

    mock_store = mock.Mock()
    mock_records = [
        DatasetRecord(
            dataset_id="dataset123",
            inputs={"question": "What is MLflow?"},
            expectations={"answer": "MLflow is..."},
        )
    ]
    mock_store._load_dataset_records.return_value = mock_records

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=mock_store):
        records = dataset.records
        assert len(records) == 1
        assert records[0].inputs["question"] == "What is MLflow?"
        mock_store._load_dataset_records.assert_called_once_with("dataset123")

        records2 = dataset.records
        assert records2 == records
        assert mock_store._load_dataset_records.call_count == 1


def test_evaluation_dataset_merge_records_validation():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    with pytest.raises(MlflowException, match="Each record must be a dictionary"):
        dataset.merge_records(["invalid", "records"])

    with pytest.raises(MlflowException, match="Each record must have an 'inputs' field"):
        dataset.merge_records([{"expectations": {"answer": "test"}}])


def test_evaluation_dataset_to_df_empty():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset._records = []

    df = dataset.to_df()
    assert len(df) == 0
    assert list(df.columns) == ["inputs", "expectations", "tags", "source_type", "source_id"]


def test_evaluation_dataset_to_df_with_records():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset._records = [
        DatasetRecord(
            dataset_record_id="rec1",
            dataset_id="dataset123",
            inputs={"question": "What is MLflow?"},
            expectations={"answer": "MLflow is a platform"},
            tags={"source": "manual"},
            source=DatasetRecordSource(source_type="HUMAN", source_data={"user_id": "user1"}),
            source_id="user1",
            created_time=123456789,
        ),
        DatasetRecord(
            dataset_record_id="rec2",
            dataset_id="dataset123",
            inputs={"question": "How to use MLflow?"},
            expectations={"answer": "You can use MLflow by..."},
            source_type="TRACE",
            source_id="trace123",
        ),
    ]

    df = dataset.to_df()
    assert len(df) == 2
    assert df.iloc[0]["inputs"]["question"] == "What is MLflow?"
    assert df.iloc[0]["expectations"]["answer"] == "MLflow is a platform"
    assert df.iloc[0]["tags"]["source"] == "manual"
    assert df.iloc[0]["source_type"] == "HUMAN"
    assert df.iloc[0]["source_id"] == "user1"
    assert df.iloc[0]["created_time"] == 123456789
    assert df.iloc[0]["dataset_record_id"] == "rec1"


def test_evaluation_dataset_proto_with_none_fields():
    dataset = EvaluationDataset(name="minimal_dataset")

    proto = dataset.to_proto()
    assert proto.name == "minimal_dataset"
    assert not proto.HasField("dataset_id")
    assert not proto.HasField("schema")
    assert not proto.HasField("profile")

    dataset2 = EvaluationDataset.from_proto(proto)
    assert dataset2.dataset_id is None
    assert dataset2.schema is None
    assert dataset2.profile is None
