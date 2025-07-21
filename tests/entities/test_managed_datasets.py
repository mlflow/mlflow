import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.managed_datasets import (
    DatasetRecord,
    DatasetRecordSource,
    DocumentSource,
    ExpectationValue,
    HumanSource,
    InputValue,
    ManagedDataset,
    TraceSource,
    create_document_source,
    create_human_source,
    create_trace_source,
    get_source_summary,
)

from tests.helper_functions import random_int, random_str


def _check_input_value(input_value, expected_key, expected_value):
    assert isinstance(input_value, InputValue)
    assert input_value.key == expected_key
    assert input_value.value == str(expected_value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("question", "What is MLflow?"),
        (
            "context",
            {"document": "mlflow.pdf", "page": 1, "metadata": ["tag1", "tag2"]},
        ),
        (
            "parameters",
            {"temperature": 0.7, "max_tokens": 100, "models": ["gpt-3.5", "gpt-4"]},
        ),
        ("input_text", "Test input for evaluation"),
        ("prompt", "Explain the concept of machine learning"),
        ("numbers", [1, 2, 3, 4, 5]),
    ],
)
def test_input_value_operations(key, value):
    input_value = InputValue(key, value)
    _check_input_value(input_value, key, value)

    proto = input_value.to_proto()
    input_value_from_proto = InputValue.from_proto(proto)
    assert input_value == input_value_from_proto
    _check_input_value(input_value_from_proto, key, value)

    expected_dict = {"key": key, "value": str(value)}
    assert input_value.to_dict() == expected_dict
    input_value_from_dict = InputValue.from_dict(expected_dict)
    _check_input_value(input_value_from_dict, key, value)


def test_input_value_equality():
    input1 = InputValue("test", "value")
    input2 = InputValue("test", "value")
    input3 = InputValue("test", "different")

    assert input1 == input2
    assert input1 != input3
    assert input1 != "not_an_input_value"


def test_input_value_string_repr():
    input_value = InputValue("question", "What is MLflow?")
    expected_repr = "<InputValue: key='question', value='What is MLflow?'>"
    assert str(input_value) == expected_repr


def _check_expectation_value(expectation, expected_value):
    assert isinstance(expectation, ExpectationValue)
    assert expectation.value == str(expected_value)


@pytest.mark.parametrize(
    "value",
    [
        "MLflow is an open-source ML lifecycle management platform",
        {"score": 4.5, "criteria": "accuracy"},
        ["positive", "informative"],
        8.5,
        True,
    ],
)
def test_expectation_value_operations(value):
    expectation = ExpectationValue(value)
    _check_expectation_value(expectation, value)

    proto = expectation.to_proto()
    expectation_from_proto = ExpectationValue.from_proto(proto)
    assert expectation == expectation_from_proto
    _check_expectation_value(expectation_from_proto, value)

    expected_dict = {"value": str(value)}
    assert expectation.to_dict() == expected_dict
    expectation_from_dict = ExpectationValue.from_dict(expected_dict)
    _check_expectation_value(expectation_from_dict, value)


@pytest.mark.parametrize(
    ("source_class", "args", "expected_attrs"),
    [
        (HumanSource, ("test_user",), {"user_id": "test_user"}),
        (
            DocumentSource,
            ("s3://bucket/document.pdf", "Document content here"),
            {"doc_uri": "s3://bucket/document.pdf", "content": "Document content here"},
        ),
        (TraceSource, (str(uuid.uuid4()), str(uuid.uuid4())), {}),
    ],
)
def test_source_creation_and_properties(source_class, args, expected_attrs):
    source = source_class(*args)

    assert isinstance(source, source_class)
    assert isinstance(source, DatasetRecordSource)

    for attr, expected_val in expected_attrs.items():
        assert getattr(source, attr) == expected_val

    if source_class == TraceSource:
        assert source.trace_id is not None
        assert source.span_id is not None


@pytest.mark.parametrize(
    ("source_class", "args"),
    [
        (HumanSource, (f"user_{random_int()}",)),
        (
            DocumentSource,
            (f"file:///path/to/{random_str()}.pdf", f"Content for {random_str()}"),
        ),
        (TraceSource, (str(uuid.uuid4()), str(uuid.uuid4()))),
    ],
)
def test_source_protobuf_conversion(source_class, args):
    source = source_class(*args)

    proto = source.to_proto()
    source_from_proto = source_class.from_proto(proto)

    assert source == source_from_proto

    if source_class == HumanSource:
        assert source_from_proto.user_id == source.user_id
    elif source_class == DocumentSource:
        assert source_from_proto.doc_uri == source.doc_uri
        assert source_from_proto.content == source.content
    elif source_class == TraceSource:
        assert source_from_proto.trace_id == source.trace_id
        assert source_from_proto.span_id == source.span_id


@pytest.mark.parametrize(
    ("source_class", "factory_func", "args"),
    [
        (HumanSource, create_human_source, ("factory_user",)),
        (DocumentSource, create_document_source, ("factory_uri", "Factory content")),
        (TraceSource, create_trace_source, (str(uuid.uuid4()), str(uuid.uuid4()))),
    ],
)
def test_source_factory_functions(source_class, factory_func, args):
    factory_source = factory_func(*args)
    direct_source = source_class(*args)

    assert isinstance(factory_source, source_class)
    assert factory_source == direct_source


def test_get_source_summary():
    human_source = HumanSource("user123")
    summary = get_source_summary(human_source)
    assert "human" in summary.lower()
    assert "user123" in summary

    doc_source = DocumentSource("path/to/doc.pdf", "content")
    summary = get_source_summary(doc_source)
    assert "document" in summary.lower()
    assert "path/to/doc.pdf" in summary

    trace_source = TraceSource("trace789", "span101")
    summary = get_source_summary(trace_source)
    assert "trace" in summary.lower()
    assert "trace789" in summary


def _create_test_record():
    inputs = [
        InputValue("question", "What is machine learning?"),
        InputValue("context", {"domain": "AI", "difficulty": "beginner"}),
    ]
    expectations = {
        "answer": ExpectationValue("Machine learning is a subset of AI..."),
        "score": ExpectationValue(8.5),
    }
    source = HumanSource("test_user")

    return DatasetRecord(
        dataset_record_id=uuid.uuid4().hex,
        dataset_id=uuid.uuid4().hex,
        inputs=inputs,
        expectations=expectations,
        source=source,
        created_by="test_user",
        last_updated_by="test_user",
        tags={"category": "qa", "difficulty": "beginner"},
    )


def test_dataset_record_creation_and_properties():
    record = _create_test_record()

    assert isinstance(record, DatasetRecord)
    assert len(record.inputs) == 2
    assert len(record.expectations) == 2
    assert isinstance(record.source, HumanSource)
    assert record.dataset_record_id is not None
    assert len(record.dataset_record_id) > 0


def test_dataset_record_protobuf_conversion():
    record = _create_test_record()

    proto = record.to_proto()
    record_from_proto = DatasetRecord.from_proto(proto)

    assert record == record_from_proto
    assert len(record_from_proto.inputs) == 2
    assert len(record_from_proto.expectations) == 2
    assert record_from_proto.dataset_record_id == record.dataset_record_id


def test_dataset_record_dict_conversion():
    record = _create_test_record()

    record_dict = record.to_dict()
    assert "inputs" in record_dict
    assert "expectations" in record_dict
    assert "source" in record_dict
    assert "dataset_record_id" in record_dict

    record_from_dict = DatasetRecord.from_dict(record_dict)
    assert record == record_from_dict


def test_dataset_record_helper_methods():
    record = _create_test_record()

    question_value = record.get_input_value("question")
    assert question_value == "What is machine learning?"

    context_value = record.get_input_value("context")
    assert isinstance(context_value, str)
    assert "domain" in context_value
    assert "AI" in context_value

    assert record.get_input_value("nonexistent") is None

    answer_expectation = record.expectations["answer"]
    assert isinstance(answer_expectation, ExpectationValue)
    assert "Machine learning" in answer_expectation.value

    score_expectation = record.expectations["score"]
    assert isinstance(score_expectation, ExpectationValue)
    assert score_expectation.value == "8.5"


def test_dataset_record_minimal_creation():
    inputs = [InputValue("input", "test")]
    record = DatasetRecord(
        dataset_record_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        inputs=inputs,
        expectations={},
    )

    assert len(record.inputs) == 1
    assert len(record.expectations) == 0
    assert record.source is None
    assert record.dataset_record_id is not None


def _create_test_dataset(with_records=True):
    dataset_id = uuid.uuid4().hex
    name = f"test_dataset_{random_str()}"
    experiment_ids = [str(random_int()), str(random_int())]

    records = []
    if with_records:
        record1 = DatasetRecord(
            dataset_record_id=uuid.uuid4().hex,
            dataset_id=dataset_id,
            inputs=[InputValue("q1", "First question")],
            expectations={"a1": ExpectationValue("First answer")},
            source=HumanSource("user1"),
            created_by="test_user",
            last_updated_by="test_user",
            tags={"category": "test"},
        )
        record2 = DatasetRecord(
            dataset_record_id=uuid.uuid4().hex,
            dataset_id=dataset_id,
            inputs=[InputValue("q2", "Second question")],
            expectations={"a2": ExpectationValue("Second answer")},
            source=HumanSource("user2"),
            created_by="test_user",
            last_updated_by="test_user",
            tags={"category": "test"},
        )
        records = [record1, record2]

    return ManagedDataset(
        dataset_id=dataset_id,
        name=name,
        experiment_ids=experiment_ids,
        records=records,
        source_type="human",
        source="test_source",
        digest="abc123",
        schema='{"type": "object"}',
        profile='{"count": 2}',
        created_by="test_user",
        last_updated_by="test_user",
    )


def _create_test_dataset_with_mixed_sources():
    dataset_id = uuid.uuid4().hex
    name = f"mixed_dataset_{random_str()}"
    experiment_ids = [str(random_int()), str(random_int())]

    human_record = DatasetRecord(
        dataset_record_id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        inputs=[InputValue("question", "What is MLflow?")],
        expectations={"answer": ExpectationValue("MLflow is an ML platform")},
        source=HumanSource("analyst123"),
        created_by="test_user",
        last_updated_by="test_user",
        tags={"type": "human_generated"},
    )

    document_record = DatasetRecord(
        dataset_record_id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        inputs=[InputValue("query", "machine learning definition")],
        expectations={"extract": ExpectationValue("ML is a subset of AI")},
        source=DocumentSource("s3://docs/ml_guide.pdf", "Machine learning content..."),
        created_by="test_user",
        last_updated_by="test_user",
        tags={"type": "document_extract"},
    )

    trace_record = DatasetRecord(
        dataset_record_id=uuid.uuid4().hex,
        dataset_id=dataset_id,
        inputs=[InputValue("prompt", "Explain neural networks")],
        expectations={"response": ExpectationValue("Neural networks are...")},
        source=TraceSource(uuid.uuid4().hex, uuid.uuid4().hex),
        created_by="test_user",
        last_updated_by="test_user",
        tags={"type": "trace_capture"},
    )

    records = [human_record, document_record, trace_record]

    return ManagedDataset(
        dataset_id=dataset_id,
        name=name,
        experiment_ids=experiment_ids,
        records=records,
        source_type="mixed",
        source="test_mixed_source",
        digest="def456",
        schema='{"type": "object", "sources": ["human", "document", "trace"]}',
        profile='{"count": 3, "source_types": 3}',
        created_by="test_user",
        last_updated_by="test_user",
    )


def test_managed_dataset_creation_and_properties():
    dataset = _create_test_dataset()

    assert isinstance(dataset, ManagedDataset)
    assert dataset.dataset_id is not None
    assert dataset.name.startswith("test_dataset_")
    assert len(dataset.experiment_ids) == 2
    assert len(dataset.records) == 2
    assert dataset.source_type == "human"
    assert dataset.source == "test_source"
    assert dataset.digest == "abc123"
    assert dataset.schema == '{"type": "object"}'
    assert dataset.profile == '{"count": 2}'
    assert dataset.created_by == "test_user"
    assert dataset.created_time is not None
    assert dataset.last_updated_time is not None


@pytest.mark.parametrize(
    ("dataset_factory", "dataset_type", "expected_record_count"),
    [
        (lambda: _create_test_dataset(), "with_records", 2),
        (lambda: _create_test_dataset(with_records=False), "without_records", 0),
        (_create_test_dataset_with_mixed_sources, "mixed_sources", 3),
    ],
)
@pytest.mark.parametrize(
    "conversion_method",
    [
        "protobuf",
        "dict",
        "json",
    ],
)
def test_dataset_conversions(
    dataset_factory, dataset_type, expected_record_count, conversion_method
):
    dataset = dataset_factory()

    if conversion_method == "protobuf":
        proto = dataset.to_proto()
        converted = ManagedDataset.from_proto(proto)
    elif conversion_method == "dict":
        data_dict = dataset.to_dict()
        if dataset_type == "with_records" or dataset_type == "mixed_sources":
            required_keys = [
                "dataset_id",
                "name",
                "experiment_ids",
                "records",
                "source_type",
                "source",
                "digest",
                "schema",
                "profile",
                "created_by",
                "created_time",
                "last_updated_time",
            ]
            for key in required_keys:
                assert key in data_dict
        converted = ManagedDataset.from_dict(data_dict)
    else:  # json
        data_dict = dataset.to_dict()
        json_str = json.dumps(data_dict)
        parsed_dict = json.loads(json_str)
        converted = ManagedDataset.from_dict(parsed_dict)

    assert dataset == converted
    assert len(converted.records) == expected_record_count
    assert converted.dataset_id == dataset.dataset_id
    assert converted.name == dataset.name

    if dataset_type == "mixed_sources":
        original_source_types = {type(r.source).__name__ for r in dataset.records}
        converted_source_types = {type(r.source).__name__ for r in converted.records}
        assert original_source_types == converted_source_types


def test_managed_dataset_create_new():
    name = "new_dataset"
    experiment_ids = ["exp1", "exp2"]

    dataset = ManagedDataset.create_new(
        name=name, experiment_ids=experiment_ids, source_type="test", created_by="creator"
    )

    assert isinstance(dataset, ManagedDataset)
    assert dataset.name == name
    assert dataset.experiment_ids == experiment_ids
    assert dataset.source_type == "test"
    assert dataset.created_by == "creator"
    assert dataset.dataset_id is not None
    assert len(dataset.records) == 0
    assert dataset.created_time is not None


def test_managed_dataset_set_profile():
    dataset = _create_test_dataset()
    original_profile = dataset.profile

    new_profile = '{"count": 5, "avg_length": 100}'
    updated_dataset = dataset.set_profile(new_profile)

    assert dataset.profile == original_profile
    assert updated_dataset.profile == new_profile
    assert updated_dataset.dataset_id == dataset.dataset_id
    assert updated_dataset.name == dataset.name
    assert updated_dataset.last_updated_time > dataset.last_updated_time


def test_managed_dataset_merge_records():
    dataset = _create_test_dataset(with_records=False)
    assert len(dataset.records) == 0

    new_records = [
        DatasetRecord(
            dataset_record_id=str(uuid.uuid4()),
            dataset_id=dataset.dataset_id,
            inputs=[InputValue("q", "Question 1")],
            expectations={"a": ExpectationValue("Answer 1")},
        ),
        DatasetRecord(
            dataset_record_id=str(uuid.uuid4()),
            dataset_id=dataset.dataset_id,
            inputs=[InputValue("q", "Question 2")],
            expectations={"a": ExpectationValue("Answer 2")},
        ),
    ]

    updated_dataset = dataset.merge_records(new_records)

    assert len(dataset.records) == 0
    assert len(updated_dataset.records) == 2
    assert updated_dataset.last_updated_time > dataset.last_updated_time


def test_managed_dataset_merge_records_with_dicts():
    dataset = _create_test_dataset(with_records=False)

    dict_records = [
        {"inputs": {"question": "What is AI?"}, "expectations": {"answer": "AI is..."}},
        {"inputs": {"question": "What is ML?"}, "expectations": {"answer": "ML is..."}},
    ]

    updated_dataset = dataset.merge_records(dict_records)
    assert len(updated_dataset.records) == 2

    assert updated_dataset.records[0].get_input_value("question") == "What is AI?"
    assert updated_dataset.records[1].get_input_value("question") == "What is ML?"


def test_managed_dataset_to_df():
    """Test that to_df() creates DataFrame with proper nested structure."""
    # Create test dataset with varied data
    inputs1 = [
        InputValue("question", "What is MLflow?"),
        InputValue("context", "MLflow is a platform..."),
    ]
    expectations1 = {
        "answer": ExpectationValue("MLflow is an open-source platform"),
        "score": ExpectationValue(8.5),
    }
    tags1 = {"category": "qa", "difficulty": "easy"}

    inputs2 = [
        InputValue("prompt", "Explain machine learning"),
        InputValue("temperature", 0.7),
    ]
    expectations2 = {
        "response": ExpectationValue("Machine learning is..."),
        "rating": ExpectationValue("excellent"),
    }
    tags2 = {"category": "explanation", "topic": "ml"}

    source1 = HumanSource("user123")
    source2 = DocumentSource("doc.pdf", "Sample content")

    record1 = DatasetRecord(
        dataset_record_id="record1",
        dataset_id="dataset1",
        inputs=inputs1,
        expectations=expectations1,
        tags=tags1,
        source=source1,
        created_time=1000000,
        last_update_time=1000001,
        created_by="user1",
        last_updated_by="user1",
    )

    record2 = DatasetRecord(
        dataset_record_id="record2",
        dataset_id="dataset1",
        inputs=inputs2,
        expectations=expectations2,
        tags=tags2,
        source=source2,
        created_time=2000000,
        last_update_time=2000001,
        created_by="user2",
        last_updated_by="user2",
    )

    dataset = ManagedDataset(
        dataset_id="dataset1",
        name="test_dataset",
        source="test_source",
        source_type="human",
        created_time=1000000,
        last_update_time=2000001,
        created_by="user1",
        last_updated_by="user2",
        experiment_ids=["exp1"],
        records=[record1, record2],
    )

    df = dataset.to_df()

    assert len(df) == 2
    expected_columns = {
        "dataset_record_id",
        "inputs",
        "expectations",
        "tags",
        "source",
        "created_time",
        "last_update_time",
        "created_by",
        "last_updated_by",
    }
    assert set(df.columns) == expected_columns

    row1 = df.iloc[0]
    assert row1["dataset_record_id"] == "record1"
    assert row1["inputs"] == {"question": "What is MLflow?", "context": "MLflow is a platform..."}
    assert row1["expectations"] == {"answer": "MLflow is an open-source platform", "score": "8.5"}
    assert row1["tags"] == {"category": "qa", "difficulty": "easy"}
    assert row1["source"]["source_type"] == "human"
    assert row1["source"]["source_data"]["user_id"] == "user123"
    assert row1["created_time"] == 1000000
    assert row1["created_by"] == "user1"

    row2 = df.iloc[1]
    assert row2["dataset_record_id"] == "record2"
    assert row2["inputs"] == {"prompt": "Explain machine learning", "temperature": "0.7"}
    assert row2["expectations"] == {"response": "Machine learning is...", "rating": "excellent"}
    assert row2["tags"] == {"category": "explanation", "topic": "ml"}
    assert row2["source"]["source_type"] == "document"
    assert row2["source"]["source_data"]["doc_uri"] == "doc.pdf"
    assert row2["created_time"] == 2000000
    assert row2["created_by"] == "user2"


def test_managed_dataset_to_df_empty():
    """Test that to_df() handles empty datasets correctly."""
    dataset = ManagedDataset(
        dataset_id="empty_dataset",
        name="empty",
        source="test",
        source_type="human",
        created_time=1000000,
        last_update_time=1000000,
        created_by="user1",
        last_updated_by="user1",
        experiment_ids=[],
        records=[],
    )

    df = dataset.to_df()
    assert len(df) == 0
    assert list(df.columns) == []


def test_managed_dataset_to_df_pandas_import_error():
    """Test that to_df() raises proper error when pandas is not available."""
    dataset = _create_test_dataset()

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="pandas is required to convert to DataFrame"):
            dataset.to_df()


def test_managed_dataset_to_df_round_trip_compatibility():
    """Test that to_df() output is compatible with merge_records() for round-trip."""
    original_dataset = _create_test_dataset()

    df = original_dataset.to_df()

    new_dataset = ManagedDataset.create_new(
        name="round_trip_test",
        experiment_ids=["exp1"],
        created_by="test_user",
    )

    df_records = df.to_dict("records")
    updated_dataset = new_dataset.merge_records(df_records)

    assert len(updated_dataset.records) == len(original_dataset.records)

    merged_df = updated_dataset.to_df()
    assert len(merged_df) == len(df)


def test_managed_dataset_from_df():
    mock_df = MagicMock()
    row1 = MagicMock()
    row1.__getitem__ = lambda self, key: {"question": "Q1", "answer": "A1"}[key]
    row1.index = ["question", "answer"]

    row2 = MagicMock()
    row2.__getitem__ = lambda self, key: {"question": "Q2", "answer": "A2"}[key]
    row2.index = ["question", "answer"]

    mock_df.iterrows.return_value = [(0, row1), (1, row2)]

    name = "df_dataset"
    experiment_ids = ["exp1"]

    dataset = ManagedDataset.from_df(
        df=mock_df,
        name=name,
        experiment_ids=experiment_ids,
        input_columns=["question"],
        expectation_columns=["answer"],
    )

    assert isinstance(dataset, ManagedDataset)
    assert dataset.name == name
    assert dataset.experiment_ids == experiment_ids
    assert len(dataset.records) == 2


def test_managed_dataset_minimal_creation():
    dataset = ManagedDataset(
        dataset_id="minimal_id", name="minimal_dataset", experiment_ids=["exp1"]
    )

    assert dataset.dataset_id == "minimal_id"
    assert dataset.name == "minimal_dataset"
    assert dataset.experiment_ids == ["exp1"]
    assert len(dataset.records) == 0
    assert dataset.source_type is None
    assert dataset.source is None
    assert dataset.digest is None


def test_managed_dataset_digest_computation():
    dataset1 = _create_test_dataset()
    dataset2 = _create_test_dataset()

    dataset1_no_digest = ManagedDataset(
        dataset_id="id1", name="dataset1", experiment_ids=["exp1"], records=dataset1.records[:1]
    )
    assert dataset1_no_digest.digest is None

    dataset2_no_digest = ManagedDataset(
        dataset_id="id2", name="dataset2", experiment_ids=["exp1"], records=dataset2.records[:1]
    )
    assert dataset2_no_digest.digest is None


def test_managed_dataset_string_repr():
    dataset = ManagedDataset(
        dataset_id="test_id", name="test_dataset", experiment_ids=["exp1", "exp2"]
    )

    repr_str = str(dataset)
    assert "ManagedDataset" in repr_str
    assert "test_id" in repr_str
    assert "test_dataset" in repr_str


def test_managed_dataset_json_serialization():
    dataset = _create_test_dataset()

    dataset_dict = dataset.to_dict()
    json_str = json.dumps(dataset_dict)

    parsed_dict = json.loads(json_str)
    restored_dataset = ManagedDataset.from_dict(parsed_dict)

    assert restored_dataset == dataset


def test_managed_dataset_with_mixed_sources_creation():
    dataset = _create_test_dataset_with_mixed_sources()

    assert isinstance(dataset, ManagedDataset)
    assert len(dataset.records) == 3
    assert dataset.source_type == "mixed"

    sources = [record.source for record in dataset.records]
    source_types = [type(source).__name__ for source in sources]

    assert "HumanSource" in source_types
    assert "DocumentSource" in source_types
    assert "TraceSource" in source_types

    human_record = next(r for r in dataset.records if isinstance(r.source, HumanSource))
    assert human_record.source.user_id == "analyst123"

    doc_record = next(r for r in dataset.records if isinstance(r.source, DocumentSource))
    assert doc_record.source.doc_uri == "s3://docs/ml_guide.pdf"
    assert doc_record.source.content == "Machine learning content..."

    trace_record = next(r for r in dataset.records if isinstance(r.source, TraceSource))
    assert trace_record.source.trace_id is not None
    assert trace_record.source.span_id is not None


@pytest.mark.parametrize(
    ("entity_class", "invalid_args", "expected_exception"),
    [
        (InputValue, (None, "value"), (ValueError, TypeError)),
        (DatasetRecord, {"inputs": "not_a_list"}, (ValueError, TypeError)),
        (
            ManagedDataset,
            {"dataset_id": "test", "name": "test", "experiment_ids": "not_a_list"},
            (ValueError, TypeError),
        ),
    ],
)
def test_invalid_entity_creation(entity_class, invalid_args, expected_exception):
    if isinstance(invalid_args, dict):
        with pytest.raises(expected_exception):
            entity_class(**invalid_args)
    else:
        with pytest.raises(expected_exception):
            entity_class(*invalid_args)


def test_protobuf_conversion_errors():
    invalid_input_proto = MagicMock()
    invalid_input_proto.key = None
    invalid_input_proto.value = "test_value"

    input_value = InputValue.from_proto(invalid_input_proto)
    assert input_value.key is None or input_value.key == ""

    invalid_expectation_proto = MagicMock()
    invalid_expectation_proto.value = None

    expectation_value = ExpectationValue.from_proto(invalid_expectation_proto)
    assert expectation_value.value is None or expectation_value.value == ""

    invalid_source_proto = MagicMock()
    invalid_source_proto.WhichOneof.return_value = None

    with pytest.raises((ValueError, AttributeError, TypeError)):
        DatasetRecordSource.from_proto(invalid_source_proto)

    invalid_source_proto.WhichOneof.return_value = "unknown_source"

    with pytest.raises((ValueError, AttributeError, KeyError)):
        DatasetRecordSource.from_proto(invalid_source_proto)

    invalid_record_proto = MagicMock()
    invalid_record_proto.inputs = [None]
    invalid_record_proto.expectations = {}
    invalid_record_proto.tags = {}
    invalid_record_proto.HasField.return_value = False

    with pytest.raises((AttributeError, TypeError)):
        DatasetRecord.from_proto(invalid_record_proto)

    invalid_record_proto2 = MagicMock()
    invalid_input_mock = MagicMock()
    invalid_input_mock.key = "test"
    invalid_input_mock.value = "test"
    invalid_record_proto2.inputs = [invalid_input_mock]

    invalid_expectations = {"key": None}
    invalid_record_proto2.expectations = invalid_expectations
    invalid_record_proto2.tags = {}
    invalid_record_proto2.HasField.return_value = False

    with pytest.raises((AttributeError, TypeError)):
        DatasetRecord.from_proto(invalid_record_proto2)

    invalid_dataset_proto = MagicMock()
    invalid_dataset_proto.dataset_id = "test_id"
    invalid_dataset_proto.name = "test_name"
    invalid_dataset_proto.experiment_ids = ["exp1"]
    invalid_dataset_proto.records = []

    def mock_has_field(field_name):
        return field_name in ["created_time"]

    invalid_dataset_proto.HasField = mock_has_field
    invalid_timestamp = MagicMock()
    invalid_timestamp.ToMilliseconds.side_effect = AttributeError("Invalid timestamp")
    invalid_dataset_proto.created_time = invalid_timestamp

    with pytest.raises(AttributeError, match="Invalid timestamp"):
        ManagedDataset.from_proto(invalid_dataset_proto)


@pytest.mark.parametrize(
    ("entity_class", "invalid_dict"),
    [
        (ManagedDataset, {"invalid": "data"}),
        (DatasetRecord, {"missing_required": "fields"}),
        (InputValue, {"no_key_or_value": True}),
        (ExpectationValue, {"missing_value": "field"}),
    ],
)
def test_dict_conversion_errors(entity_class, invalid_dict):
    with pytest.raises((KeyError, ValueError, TypeError)):
        entity_class.from_dict(invalid_dict)


@pytest.mark.parametrize(
    "invalid_records",
    [
        "not_a_list_or_dataframe",
        123,
        None,
        {"not": "a_list"},
    ],
)
def test_merge_records_type_errors(invalid_records):
    dataset = ManagedDataset(dataset_id="test", name="test", experiment_ids=["exp1"])

    with pytest.raises((ValueError, TypeError)):
        dataset.merge_records(invalid_records)
