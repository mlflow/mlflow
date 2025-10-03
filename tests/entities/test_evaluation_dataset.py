import pandas as pd

from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.evaluation_dataset import EvaluationDataset


def test_evaluation_dataset_creation():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=987654321,
        tags={"source": "manual", "type": "HUMAN"},
        schema='{"fields": ["input", "output"]}',
        profile='{"count": 100}',
        created_by="user1",
        last_updated_by="user2",
    )

    assert dataset.dataset_id == "dataset123"
    assert dataset.name == "test_dataset"
    assert dataset.tags == {"source": "manual", "type": "HUMAN"}
    assert dataset.schema == '{"fields": ["input", "output"]}'
    assert dataset.profile == '{"count": 100}'
    assert dataset.digest == "abc123"
    assert dataset.created_by == "user1"
    assert dataset.last_updated_by == "user2"
    assert dataset.created_time == 123456789
    assert dataset.last_update_time == 987654321

    dataset.experiment_ids = ["exp1", "exp2"]
    assert dataset.experiment_ids == ["exp1", "exp2"]


def test_evaluation_dataset_timestamps_required():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=987654321,
    )

    assert dataset.created_time == 123456789
    assert dataset.last_update_time == 987654321


def test_evaluation_dataset_experiment_ids_setter():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    new_experiment_ids = ["exp1", "exp2"]
    dataset.experiment_ids = new_experiment_ids
    assert dataset._experiment_ids == new_experiment_ids
    assert dataset.experiment_ids == new_experiment_ids

    dataset.experiment_ids = []
    assert dataset._experiment_ids == []
    assert dataset.experiment_ids == []

    dataset.experiment_ids = None
    assert dataset._experiment_ids == []
    assert dataset.experiment_ids == []


def test_evaluation_dataset_to_from_proto():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        tags={"source": "manual", "type": "HUMAN"},
        schema='{"fields": ["input", "output"]}',
        profile='{"count": 100}',
        digest="abc123",
        created_time=123456789,
        last_update_time=987654321,
        created_by="user1",
        last_updated_by="user2",
    )
    dataset.experiment_ids = ["exp1", "exp2"]

    proto = dataset.to_proto()
    assert proto.name == "test_dataset"
    assert proto.tags == '{"source": "manual", "type": "HUMAN"}'
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
    assert dataset2.tags == dataset.tags
    assert dataset2.schema == dataset.schema
    assert dataset2.profile == dataset.profile
    assert dataset2.digest == dataset.digest
    assert dataset2.created_time == dataset.created_time
    assert dataset2.last_update_time == dataset.last_update_time
    assert dataset2.created_by == dataset.created_by
    assert dataset2.last_updated_by == dataset.last_updated_by
    assert dataset2._experiment_ids == ["exp1", "exp2"]
    assert dataset2.experiment_ids == ["exp1", "exp2"]


def test_evaluation_dataset_to_from_proto_minimal():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    proto = dataset.to_proto()
    dataset2 = EvaluationDataset.from_proto(proto)

    assert dataset2.dataset_id == "dataset123"
    assert dataset2.name == "test_dataset"
    assert dataset2.tags is None
    assert dataset2.schema is None
    assert dataset2.profile is None
    assert dataset2.digest == "digest123"
    assert dataset2.created_by is None
    assert dataset2.last_updated_by is None
    assert dataset2._experiment_ids is None


def test_evaluation_dataset_to_from_dict():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        tags={"source": "manual", "type": "HUMAN"},
        schema='{"fields": ["input", "output"]}',
        profile='{"count": 100}',
        digest="abc123",
        created_time=123456789,
        last_update_time=987654321,
        created_by="user1",
        last_updated_by="user2",
    )
    dataset.experiment_ids = ["exp1", "exp2"]

    dataset._records = [
        DatasetRecord(
            dataset_record_id="rec789",
            dataset_id="dataset123",
            inputs={"question": "What is MLflow?"},
            created_time=123456789,
            last_update_time=123456789,
        )
    ]

    data = dataset.to_dict()
    assert data["dataset_id"] == "dataset123"
    assert data["name"] == "test_dataset"
    assert data["tags"] == {"source": "manual", "type": "HUMAN"}
    assert data["schema"] == '{"fields": ["input", "output"]}'
    assert data["profile"] == '{"count": 100}'
    assert data["digest"] == "abc123"
    assert data["created_time"] == 123456789
    assert data["last_update_time"] == 987654321
    assert data["created_by"] == "user1"
    assert data["last_updated_by"] == "user2"
    assert data["experiment_ids"] == ["exp1", "exp2"]
    assert len(data["records"]) == 1
    assert data["records"][0]["inputs"]["question"] == "What is MLflow?"

    dataset2 = EvaluationDataset.from_dict(data)
    assert dataset2.dataset_id == dataset.dataset_id
    assert dataset2.name == dataset.name
    assert dataset2.tags == dataset.tags
    assert dataset2.schema == dataset.schema
    assert dataset2.profile == dataset.profile
    assert dataset2.digest == dataset.digest
    assert dataset2.created_time == dataset.created_time
    assert dataset2.last_update_time == dataset.last_update_time
    assert dataset2.created_by == dataset.created_by
    assert dataset2.last_updated_by == dataset.last_updated_by
    assert dataset2._experiment_ids == ["exp1", "exp2"]
    assert dataset2.experiment_ids == ["exp1", "exp2"]
    assert len(dataset2._records) == 1
    assert dataset2._records[0].inputs["question"] == "What is MLflow?"


def test_evaluation_dataset_to_from_dict_minimal():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )
    dataset._experiment_ids = []
    dataset._records = []

    data = dataset.to_dict()
    dataset2 = EvaluationDataset.from_dict(data)

    assert dataset2.dataset_id == "dataset123"
    assert dataset2.name == "test_dataset"
    assert dataset2.tags is None
    assert dataset2.schema is None
    assert dataset2.profile is None
    assert dataset2.digest == "digest123"
    assert dataset2.created_by is None
    assert dataset2.last_updated_by is None
    assert dataset2._experiment_ids == []
    assert dataset2._records == []


def test_evaluation_dataset_has_records():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    assert dataset.has_records() is False

    dataset._records = [
        DatasetRecord(
            dataset_record_id="rec123",
            dataset_id="dataset123",
            inputs={"test": "data"},
            created_time=123456789,
            last_update_time=123456789,
        )
    ]
    assert dataset.has_records() is True

    dataset._records = []
    assert dataset.has_records() is True


def test_evaluation_dataset_proto_with_unloaded_experiment_ids():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    assert dataset._experiment_ids is None

    proto = dataset.to_proto()
    assert len(proto.experiment_ids) == 0
    assert dataset._experiment_ids is None


def test_evaluation_dataset_complex_tags():
    complex_tags = {
        "source": "automated",
        "metadata": {"version": "1.0", "config": {"temperature": 0.7, "max_tokens": 100}},
        "labels": ["production", "evaluated"],
    }

    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
        tags=complex_tags,
    )

    proto = dataset.to_proto()
    dataset2 = EvaluationDataset.from_proto(proto)
    assert dataset2.tags == complex_tags

    dataset._experiment_ids = []
    dataset._records = []

    data = dataset.to_dict()
    dataset3 = EvaluationDataset.from_dict(data)
    assert dataset3.tags == complex_tags


def test_evaluation_dataset_to_df():
    """Test that to_df method returns a DataFrame with outputs column."""
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    # Test empty dataset
    df_empty = dataset.to_df()
    assert isinstance(df_empty, pd.DataFrame)
    expected_columns = [
        "inputs",
        "outputs",
        "expectations",
        "tags",
        "source_type",
        "source_id",
        "created_time",
        "dataset_record_id",
    ]
    assert list(df_empty.columns) == expected_columns
    assert len(df_empty) == 0

    # Test dataset with records
    dataset._records = [
        DatasetRecord(
            dataset_record_id="rec123",
            dataset_id="dataset123",
            inputs={"question": "What is MLflow?"},
            outputs={
                "answer": "MLflow is an ML platform for managing machine learning lifecycle",
                "key1": "value1",
            },
            expectations={"answer": "MLflow is an ML platform"},
            tags={"source": "manual"},
            source_type="HUMAN",
            source_id="user123",
            created_time=123456789,
            last_update_time=123456789,
        ),
        DatasetRecord(
            dataset_record_id="rec456",
            dataset_id="dataset123",
            inputs={"question": "What is Spark?"},
            outputs={"answer": "Apache Spark is a unified analytics engine for data processing"},
            expectations={"answer": "Spark is a data engine"},
            tags={"source": "automated"},
            source_type="CODE",
            source_id="script456",
            created_time=123456790,
            last_update_time=123456790,
        ),
    ]

    df = dataset.to_df()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns
    assert len(df) == 2

    # Check that outputs column exists and contains actual values
    assert "outputs" in df.columns
    assert df["outputs"].iloc[0] == {
        "answer": "MLflow is an ML platform for managing machine learning lifecycle",
        "key1": "value1",
    }
    assert df["outputs"].iloc[1] == {
        "answer": "Apache Spark is a unified analytics engine for data processing"
    }

    # Check other columns have expected values
    assert df["inputs"].iloc[0] == {"question": "What is MLflow?"}
    assert df["inputs"].iloc[1] == {"question": "What is Spark?"}
    assert df["expectations"].iloc[0] == {"answer": "MLflow is an ML platform"}
    assert df["expectations"].iloc[1] == {"answer": "Spark is a data engine"}
    assert df["tags"].iloc[0] == {"source": "manual"}
    assert df["tags"].iloc[1] == {"source": "automated"}
    assert df["source_type"].iloc[0] == "HUMAN"
    assert df["source_type"].iloc[1] == "CODE"
    assert df["source_id"].iloc[0] == "user123"
    assert df["source_id"].iloc[1] == "script456"
    assert df["dataset_record_id"].iloc[0] == "rec123"
    assert df["dataset_record_id"].iloc[1] == "rec456"
