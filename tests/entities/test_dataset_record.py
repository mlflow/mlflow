import json

import pytest

from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.dataset_record_source import DatasetRecordSource
from mlflow.protos.datasets_pb2 import DatasetRecord as ProtoDatasetRecord
from mlflow.protos.datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource


def test_dataset_record_creation():
    source = DatasetRecordSource(
        source_type="HUMAN", source_data={"user_id": "user1", "timestamp": "2024-01-01"}
    )
    record = DatasetRecord(
        dataset_record_id="rec123",
        dataset_id="dataset123",
        inputs={"question": "What is MLflow?", "context": "MLflow is a platform"},
        created_time=123456789,
        last_update_time=987654321,
        expectations={"answer": "MLflow is an open source platform"},
        tags={"source": "manual", "quality": "high"},
        source=source,
        source_id="user1",
        source_type="HUMAN",
        created_by="user1",
        last_updated_by="user2",
    )

    assert record.dataset_record_id == "rec123"
    assert record.dataset_id == "dataset123"
    assert record.inputs == {"question": "What is MLflow?", "context": "MLflow is a platform"}
    assert record.expectations == {"answer": "MLflow is an open source platform"}
    assert record.tags == {"source": "manual", "quality": "high"}
    assert record.source.source_type == "HUMAN"
    assert record.source.source_data["user_id"] == "user1"
    assert record.source_id == "user1"
    assert record.source_type == "HUMAN"
    assert record.created_by == "user1"
    assert record.last_updated_by == "user2"


def test_dataset_record_empty_inputs_validation():
    # Empty dict is allowed (for traces without inputs)
    record = DatasetRecord(
        dataset_record_id="rec123",
        dataset_id="dataset123",
        inputs={},
        created_time=123456789,
        last_update_time=123456789,
    )
    assert record.inputs == {}

    # None is not allowed
    with pytest.raises(ValueError, match="inputs must be provided"):
        DatasetRecord(
            dataset_record_id="rec123",
            dataset_id="dataset123",
            inputs=None,
            created_time=123456789,
            last_update_time=123456789,
        )


@pytest.mark.parametrize(
    (
        "source_type",
        "source_data",
        "explicit_source_id",
        "explicit_source_type",
        "expected_source_id",
        "expected_source_type",
    ),
    [
        ("TRACE", {"trace_id": "trace123", "span_id": "span456"}, None, None, "trace123", "TRACE"),
        (
            "DOCUMENT",
            {"source_id": "doc123", "doc_uri": "s3://bucket/doc.txt"},
            None,
            None,
            "doc123",
            "DOCUMENT",
        ),
        ("HUMAN", {"source_id": "human123", "user_id": "user1"}, None, None, "human123", "HUMAN"),
        ("CODE", {"source_id": "code123", "function": "evaluate"}, None, None, "code123", "CODE"),
        ("TRACE", {"trace_id": "trace123"}, "override123", None, "override123", "TRACE"),
        ("HUMAN", {"user_id": "user1"}, None, "CUSTOM_TYPE", None, "CUSTOM_TYPE"),
        ("TRACE", {"some_other_key": "value"}, None, None, None, "TRACE"),
    ],
)
def test_dataset_record_source_id_and_type_extraction(
    source_type,
    source_data,
    explicit_source_id,
    explicit_source_type,
    expected_source_id,
    expected_source_type,
):
    kwargs = {
        "dataset_record_id": "rec123",
        "dataset_id": "dataset123",
        "inputs": {"test": "data"},
        "created_time": 123456789,
        "last_update_time": 123456789,
        "source": DatasetRecordSource(source_type=source_type, source_data=source_data),
    }
    if explicit_source_id is not None:
        kwargs["source_id"] = explicit_source_id
    if explicit_source_type is not None:
        kwargs["source_type"] = explicit_source_type

    record = DatasetRecord(**kwargs)
    assert record.source_id == expected_source_id
    assert record.source_type == expected_source_type


def test_dataset_record_to_from_proto():
    record = DatasetRecord(
        dataset_record_id="rec123",
        dataset_id="dataset123",
        inputs={"question": "What is MLflow?"},
        expectations={"answer": "MLflow is a platform"},
        tags={"source": "manual"},
        source=DatasetRecordSource(source_type="HUMAN", source_data={"user_id": "user1"}),
        source_id="user1",
        source_type="HUMAN",
        created_time=123456789,
        last_update_time=987654321,
        created_by="user1",
        last_updated_by="user2",
    )

    proto = record.to_proto()
    assert isinstance(proto, ProtoDatasetRecord)
    assert proto.dataset_record_id == "rec123"
    assert proto.dataset_id == "dataset123"
    assert json.loads(proto.inputs) == {"question": "What is MLflow?"}
    assert json.loads(proto.expectations) == {"answer": "MLflow is a platform"}
    assert json.loads(proto.tags) == {"source": "manual"}
    assert json.loads(proto.source) == {"source_type": "HUMAN", "source_data": {"user_id": "user1"}}
    assert proto.source_id == "user1"
    assert proto.source_type == ProtoDatasetRecordSource.SourceType.Value("HUMAN")
    assert proto.created_time == 123456789
    assert proto.last_update_time == 987654321
    assert proto.created_by == "user1"
    assert proto.last_updated_by == "user2"

    record2 = DatasetRecord.from_proto(proto)
    assert record2.dataset_record_id == record.dataset_record_id
    assert record2.dataset_id == record.dataset_id
    assert record2.inputs == record.inputs
    assert record2.expectations == record.expectations
    assert record2.tags == record.tags
    assert record2.source == record.source
    assert record2.source_id == record.source_id
    assert record2.source_type == record.source_type
    assert record2.created_time == record.created_time
    assert record2.last_update_time == record.last_update_time
    assert record2.created_by == record.created_by
    assert record2.last_updated_by == record.last_updated_by


def test_dataset_record_to_from_proto_with_none_values():
    record = DatasetRecord(
        dataset_id="dataset123",
        inputs={"question": "test"},
        dataset_record_id="rec123",
        created_time=123456789,
        last_update_time=123456789,
    )

    proto = record.to_proto()
    record2 = DatasetRecord.from_proto(proto)

    assert record2.dataset_record_id == "rec123"
    assert record2.dataset_id == "dataset123"
    assert record2.inputs == {"question": "test"}
    assert record2.expectations is None
    assert record2.tags == {}
    assert record2.source is None


def test_dataset_record_to_from_dict():
    record = DatasetRecord(
        dataset_record_id="rec123",
        dataset_id="dataset123",
        inputs={"question": "What is MLflow?"},
        expectations={"answer": "MLflow is a platform"},
        tags={"source": "manual"},
        source=DatasetRecordSource(source_type="HUMAN", source_data={"user_id": "user1"}),
        source_id="user1",
        source_type="HUMAN",
        created_time=123456789,
        last_update_time=987654321,
        created_by="user1",
        last_updated_by="user2",
    )

    data = record.to_dict()
    assert data["dataset_record_id"] == "rec123"
    assert data["dataset_id"] == "dataset123"
    assert data["inputs"] == {"question": "What is MLflow?"}
    assert data["expectations"] == {"answer": "MLflow is a platform"}
    assert data["tags"] == {"source": "manual"}
    assert data["source"] == {"source_type": "HUMAN", "source_data": {"user_id": "user1"}}
    assert data["source_id"] == "user1"
    assert data["source_type"] == "HUMAN"
    assert data["created_time"] == 123456789
    assert data["last_update_time"] == 987654321
    assert data["created_by"] == "user1"
    assert data["last_updated_by"] == "user2"

    record2 = DatasetRecord.from_dict(data)
    assert record2 == record


def test_dataset_record_equality():
    source = DatasetRecordSource(source_type="HUMAN", source_data={"user_id": "user1"})
    record1 = DatasetRecord(
        dataset_record_id="rec123",
        dataset_id="dataset123",
        inputs={"question": "What is MLflow?"},
        created_time=123456789,
        last_update_time=123456789,
        expectations={"answer": "MLflow is a platform"},
        tags={"source": "manual"},
        source=source,
        source_id="user1",
        source_type="HUMAN",
    )

    record2 = DatasetRecord(
        dataset_record_id="rec123",
        dataset_id="dataset123",
        inputs={"question": "What is MLflow?"},
        created_time=123456789,
        last_update_time=123456789,
        expectations={"answer": "MLflow is a platform"},
        tags={"source": "manual"},
        source=source,
        source_id="user1",
        source_type="HUMAN",
    )

    record3 = DatasetRecord(
        dataset_record_id="rec456",
        dataset_id="dataset123",
        inputs={"question": "What is MLflow?"},
        created_time=123456789,
        last_update_time=123456789,
        expectations={"answer": "MLflow is a platform"},
        tags={"source": "manual"},
        source=source,
        source_id="user1",
        source_type="HUMAN",
    )

    assert record1 == record2
    assert record1 != record3
    assert record1 != "not a record"


@pytest.mark.parametrize(
    ("test_case", "kwargs", "expected_source", "expected_source_id", "expected_source_type"),
    [
        (
            "none_source",
            {
                "dataset_record_id": "rec123",
                "dataset_id": "dataset123",
                "inputs": {"question": "test"},
                "created_time": 123456789,
                "last_update_time": 123456789,
                "source": None,
            },
            None,
            None,
            None,
        ),
        (
            "dict_source",
            {
                "dataset_record_id": "rec456",
                "dataset_id": "dataset123",
                "inputs": {"question": "test"},
                "created_time": 123456789,
                "last_update_time": 123456789,
                "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace123"}},
            },
            {"source_type": "TRACE", "source_data": {"trace_id": "trace123"}},
            None,
            None,
        ),
        (
            "explicit_override",
            {
                "dataset_record_id": "rec789",
                "dataset_id": "dataset123",
                "inputs": {"question": "test"},
                "created_time": 123456789,
                "last_update_time": 123456789,
                "source": DatasetRecordSource(
                    source_type="TRACE", source_data={"trace_id": "trace123"}
                ),
                "source_id": "explicit_id",
                "source_type": "EXPLICIT_TYPE",
            },
            DatasetRecordSource(source_type="TRACE", source_data={"trace_id": "trace123"}),
            "explicit_id",
            "EXPLICIT_TYPE",
        ),
    ],
)
def test_dataset_record_source_edge_cases(
    test_case, kwargs, expected_source, expected_source_id, expected_source_type
):
    record = DatasetRecord(**kwargs)

    if expected_source is None:
        assert record.source is None
    elif isinstance(expected_source, dict):
        assert record.source == expected_source
    else:
        assert record.source.source_type == expected_source.source_type
        assert record.source.source_data == expected_source.source_data

    assert record.source_id == expected_source_id
    assert record.source_type == expected_source_type


def test_dataset_record_from_dict_with_missing_keys():
    # Test with all required fields present
    minimal_data = {
        "dataset_record_id": "rec123",
        "dataset_id": "dataset123",
        "inputs": {"question": "test"},
        "created_time": 123456789,
        "last_update_time": 987654321,
    }
    record = DatasetRecord.from_dict(minimal_data)

    assert record.dataset_record_id == "rec123"
    assert record.dataset_id == "dataset123"
    assert record.inputs == {"question": "test"}
    assert record.expectations is None
    assert record.tags == {}
    assert record.source is None
    assert record.source_id is None
    assert record.source_type is None
    assert record.created_time == 123456789
    assert record.last_update_time == 987654321
    assert record.created_by is None
    assert record.last_updated_by is None

    # Test missing required fields
    with pytest.raises(ValueError, match="dataset_id is required"):
        DatasetRecord.from_dict(
            {
                "dataset_record_id": "rec123",
                "inputs": {"test": "data"},
                "created_time": 123,
                "last_update_time": 123,
            }
        )

    with pytest.raises(ValueError, match="dataset_record_id is required"):
        DatasetRecord.from_dict(
            {
                "dataset_id": "dataset123",
                "inputs": {"test": "data"},
                "created_time": 123,
                "last_update_time": 123,
            }
        )

    with pytest.raises(ValueError, match="inputs is required"):
        DatasetRecord.from_dict(
            {
                "dataset_record_id": "rec123",
                "dataset_id": "dataset123",
                "created_time": 123,
                "last_update_time": 123,
            }
        )

    with pytest.raises(ValueError, match="created_time is required"):
        DatasetRecord.from_dict(
            {
                "dataset_record_id": "rec123",
                "dataset_id": "dataset123",
                "inputs": {"test": "data"},
                "last_update_time": 123,
            }
        )

    with pytest.raises(ValueError, match="last_update_time is required"):
        DatasetRecord.from_dict(
            {
                "dataset_record_id": "rec123",
                "dataset_id": "dataset123",
                "inputs": {"test": "data"},
                "created_time": 123,
            }
        )

    # Test that empty inputs dict is allowed
    record_empty_inputs = DatasetRecord.from_dict(
        {
            "dataset_record_id": "rec789",
            "dataset_id": "dataset123",
            "inputs": {},
            "created_time": 123,
            "last_update_time": 123,
        }
    )
    assert record_empty_inputs.inputs == {}

    # Test that missing inputs raises ValueError
    with pytest.raises(ValueError, match="inputs is required"):
        DatasetRecord.from_dict(
            {
                "dataset_record_id": "rec789",
                "dataset_id": "dataset123",
                "created_time": 123,
                "last_update_time": 123,
            }
        )

    data_with_source = {
        "dataset_record_id": "rec456",
        "dataset_id": "dataset456",
        "inputs": {"test": "data"},
        "created_time": 123456789,
        "last_update_time": 987654321,
        "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace123"}},
    }
    record3 = DatasetRecord.from_dict(data_with_source)

    assert record3.source.source_type == "TRACE"
    assert record3.source_id == "trace123"
    assert record3.source_type == "TRACE"


def test_dataset_record_complex_inputs():
    complex_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is MLflow?"},
        ],
        "metadata": {"temperature": 0.7, "max_tokens": 100, "model": "gpt-4"},
        "context": ["doc1", "doc2", "doc3"],
    }

    record = DatasetRecord(
        dataset_id="dataset123",
        dataset_record_id="rec123",
        inputs=complex_data,
        created_time=123456789,
        last_update_time=123456789,
        expectations={
            "response": "MLflow is an open source platform for ML lifecycle",
            "confidence": 0.95,
            "sources": ["doc1", "doc2"],
        },
    )

    proto = record.to_proto()
    record2 = DatasetRecord.from_proto(proto)

    assert record2.inputs == complex_data
    assert record2.expectations["response"] == "MLflow is an open source platform for ML lifecycle"
    assert record2.expectations["confidence"] == 0.95
    assert record2.expectations["sources"] == ["doc1", "doc2"]

    data = record.to_dict()
    record3 = DatasetRecord.from_dict(data)

    assert record3.inputs == complex_data
    assert record3.expectations == record.expectations
