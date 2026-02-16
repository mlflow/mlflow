import json

import pytest

from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.exceptions import MlflowException
from mlflow.protos.datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource


def test_dataset_record_source_type_constants():
    assert DatasetRecordSourceType.TRACE == "TRACE"
    assert DatasetRecordSourceType.HUMAN == "HUMAN"
    assert DatasetRecordSourceType.DOCUMENT == "DOCUMENT"
    assert DatasetRecordSourceType.CODE == "CODE"
    assert DatasetRecordSourceType.UNSPECIFIED == "UNSPECIFIED"


def test_dataset_record_source_type_enum_values():
    assert DatasetRecordSourceType.TRACE == "TRACE"
    assert DatasetRecordSourceType.HUMAN == "HUMAN"
    assert DatasetRecordSourceType.DOCUMENT == "DOCUMENT"
    assert DatasetRecordSourceType.CODE == "CODE"
    assert DatasetRecordSourceType.UNSPECIFIED == "UNSPECIFIED"

    assert isinstance(DatasetRecordSourceType.TRACE, str)
    assert DatasetRecordSourceType.TRACE.value == "TRACE"


def test_dataset_record_source_string_normalization():
    source1 = DatasetRecordSource(source_type="trace", source_data={})
    assert source1.source_type == DatasetRecordSourceType.TRACE

    source2 = DatasetRecordSource(source_type="HUMAN", source_data={})
    assert source2.source_type == DatasetRecordSourceType.HUMAN

    source3 = DatasetRecordSource(source_type="Document", source_data={})
    assert source3.source_type == DatasetRecordSourceType.DOCUMENT

    source4 = DatasetRecordSource(source_type=DatasetRecordSourceType.CODE, source_data={})
    assert source4.source_type == DatasetRecordSourceType.CODE


def test_dataset_record_source_invalid_type():
    with pytest.raises(MlflowException, match="Invalid dataset record source type"):
        DatasetRecordSource(source_type="INVALID", source_data={})


def test_dataset_record_source_creation():
    source1 = DatasetRecordSource(
        source_type="TRACE", source_data={"trace_id": "trace123", "span_id": "span456"}
    )

    assert source1.source_type == DatasetRecordSourceType.TRACE
    assert source1.source_data == {"trace_id": "trace123", "span_id": "span456"}

    source2 = DatasetRecordSource(
        source_type=DatasetRecordSourceType.HUMAN, source_data={"user_id": "user123"}
    )

    assert source2.source_type == DatasetRecordSourceType.HUMAN
    assert source2.source_data == {"user_id": "user123"}


def test_dataset_record_source_auto_normalization():
    source = DatasetRecordSource(source_type="trace", source_data={"trace_id": "trace123"})

    assert source.source_type == DatasetRecordSourceType.TRACE


def test_dataset_record_source_empty_data():
    source = DatasetRecordSource(source_type="HUMAN", source_data=None)
    assert source.source_data == {}


def test_trace_source():
    source1 = DatasetRecordSource(
        source_type="TRACE", source_data={"trace_id": "trace123", "span_id": "span456"}
    )
    assert source1.source_type == DatasetRecordSourceType.TRACE
    assert source1.source_data["trace_id"] == "trace123"
    assert source1.source_data.get("span_id") == "span456"

    source2 = DatasetRecordSource(
        source_type=DatasetRecordSourceType.TRACE, source_data={"trace_id": "trace789"}
    )
    assert source2.source_data["trace_id"] == "trace789"
    assert source2.source_data.get("span_id") is None


def test_human_source():
    source1 = DatasetRecordSource(source_type="HUMAN", source_data={"user_id": "user123"})
    assert source1.source_type == DatasetRecordSourceType.HUMAN
    assert source1.source_data["user_id"] == "user123"

    source2 = DatasetRecordSource(
        source_type=DatasetRecordSourceType.HUMAN,
        source_data={"user_id": "user456", "timestamp": "2024-01-01"},
    )
    assert source2.source_data["user_id"] == "user456"
    assert source2.source_data["timestamp"] == "2024-01-01"


def test_document_source():
    source1 = DatasetRecordSource(
        source_type="DOCUMENT",
        source_data={"doc_uri": "s3://bucket/doc.txt", "content": "Document content"},
    )
    assert source1.source_type == DatasetRecordSourceType.DOCUMENT
    assert source1.source_data["doc_uri"] == "s3://bucket/doc.txt"
    assert source1.source_data["content"] == "Document content"

    source2 = DatasetRecordSource(
        source_type=DatasetRecordSourceType.DOCUMENT,
        source_data={"doc_uri": "https://example.com/doc.pdf"},
    )
    assert source2.source_data["doc_uri"] == "https://example.com/doc.pdf"
    assert source2.source_data.get("content") is None


def test_dataset_record_source_to_from_proto():
    source = DatasetRecordSource(source_type="CODE", source_data={"file": "example.py", "line": 42})

    proto = source.to_proto()
    assert isinstance(proto, ProtoDatasetRecordSource)
    assert proto.source_type == ProtoDatasetRecordSource.SourceType.Value("CODE")
    assert json.loads(proto.source_data) == {"file": "example.py", "line": 42}

    source2 = DatasetRecordSource.from_proto(proto)
    assert isinstance(source2, DatasetRecordSource)
    assert source2.source_type == DatasetRecordSourceType.CODE
    assert source2.source_data == {"file": "example.py", "line": 42}


def test_trace_source_proto_conversion():
    source = DatasetRecordSource(
        source_type="TRACE", source_data={"trace_id": "trace123", "span_id": "span456"}
    )

    proto = source.to_proto()
    assert proto.source_type == ProtoDatasetRecordSource.SourceType.Value("TRACE")

    source2 = DatasetRecordSource.from_proto(proto)
    assert isinstance(source2, DatasetRecordSource)
    assert source2.source_data["trace_id"] == "trace123"
    assert source2.source_data["span_id"] == "span456"


def test_human_source_proto_conversion():
    source = DatasetRecordSource(source_type="HUMAN", source_data={"user_id": "user123"})

    proto = source.to_proto()
    assert proto.source_type == ProtoDatasetRecordSource.SourceType.Value("HUMAN")

    source2 = DatasetRecordSource.from_proto(proto)
    assert isinstance(source2, DatasetRecordSource)
    assert source2.source_data["user_id"] == "user123"


def test_document_source_proto_conversion():
    source = DatasetRecordSource(
        source_type="DOCUMENT",
        source_data={"doc_uri": "s3://bucket/doc.txt", "content": "Test content"},
    )

    proto = source.to_proto()
    assert proto.source_type == ProtoDatasetRecordSource.SourceType.Value("DOCUMENT")

    source2 = DatasetRecordSource.from_proto(proto)
    assert isinstance(source2, DatasetRecordSource)
    assert source2.source_data["doc_uri"] == "s3://bucket/doc.txt"
    assert source2.source_data["content"] == "Test content"


def test_dataset_record_source_to_from_dict():
    source = DatasetRecordSource(source_type="CODE", source_data={"file": "example.py", "line": 42})

    data = source.to_dict()
    assert data == {"source_type": "CODE", "source_data": {"file": "example.py", "line": 42}}

    source2 = DatasetRecordSource.from_dict(data)
    assert source2.source_type == DatasetRecordSourceType.CODE
    assert source2.source_data == {"file": "example.py", "line": 42}


def test_specific_source_dict_conversion():
    trace_data = {"source_type": "TRACE", "source_data": {"trace_id": "trace123"}}
    trace_source = DatasetRecordSource.from_dict(trace_data)
    assert isinstance(trace_source, DatasetRecordSource)
    assert trace_source.source_data["trace_id"] == "trace123"

    human_data = {"source_type": "HUMAN", "source_data": {"user_id": "user123"}}
    human_source = DatasetRecordSource.from_dict(human_data)
    assert isinstance(human_source, DatasetRecordSource)
    assert human_source.source_data["user_id"] == "user123"

    doc_data = {"source_type": "DOCUMENT", "source_data": {"doc_uri": "file.txt"}}
    doc_source = DatasetRecordSource.from_dict(doc_data)
    assert isinstance(doc_source, DatasetRecordSource)
    assert doc_source.source_data["doc_uri"] == "file.txt"


def test_dataset_record_source_equality():
    source1 = DatasetRecordSource(source_type="TRACE", source_data={"trace_id": "trace123"})

    source2 = DatasetRecordSource(source_type="TRACE", source_data={"trace_id": "trace123"})

    source3 = DatasetRecordSource(source_type="TRACE", source_data={"trace_id": "trace456"})

    source4 = DatasetRecordSource(source_type="HUMAN", source_data={"trace_id": "trace123"})

    assert source1 == source2
    assert source1 != source3
    assert source1 != source4
    assert source1 != "not a source"


def test_dataset_record_source_with_extra_fields():
    source = DatasetRecordSource(
        source_type="HUMAN",
        source_data={
            "user_id": "user123",
            "timestamp": "2024-01-01T00:00:00Z",
            "annotation_tool": "labelstudio",
            "confidence": 0.95,
        },
    )

    assert source.source_data["user_id"] == "user123"
    assert source.source_data["timestamp"] == "2024-01-01T00:00:00Z"
    assert source.source_data["annotation_tool"] == "labelstudio"
    assert source.source_data["confidence"] == 0.95

    proto = source.to_proto()
    source2 = DatasetRecordSource.from_proto(proto)
    assert source2.source_data == source.source_data
