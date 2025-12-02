import json

import pandas as pd
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.dataset_record_source import DatasetRecordSourceType
from mlflow.entities.evaluation_dataset import EvaluationDataset
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils import build_otel_context


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


def test_evaluation_dataset_to_df(monkeypatch):
    """Test that to_df method returns a DataFrame with outputs column."""

    class DummyTrackingStore:
        def _load_dataset_records(self, dataset_id, max_results=None):
            return [], None

    dummy_store = DummyTrackingStore()
    monkeypatch.setattr(
        "mlflow.tracking._tracking_service.utils._get_store",
        lambda: dummy_store,
    )

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


def create_test_span(
    span_id=1,
    parent_id=None,
    name="test_span",
    inputs=None,
    outputs=None,
    span_type=SpanType.UNKNOWN,
):
    attributes = {
        "mlflow.spanType": json.dumps(span_type),
    }

    if inputs is not None:
        attributes["mlflow.spanInputs"] = json.dumps(inputs)

    if outputs is not None:
        attributes["mlflow.spanOutputs"] = json.dumps(outputs)

    otel_span = OTelReadableSpan(
        name=name,
        context=build_otel_context(trace_id=123456789, span_id=span_id),
        parent=build_otel_context(trace_id=123456789, span_id=parent_id) if parent_id else None,
        start_time=100000000,
        end_time=200000000,
        attributes=attributes,
    )
    return Span(otel_span)


def create_test_trace(
    trace_id="test-trace-123", inputs=None, outputs=None, expectations=None, _no_defaults=False
):
    assessments = []
    if expectations:
        from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType, Expectation

        for name, value in expectations.items():
            expectation = Expectation(
                name=name,
                value=value,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN, source_id="test_user"
                ),
            )
            assessments.append(expectation)

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        execution_duration=1000,
        state=TraceState.OK,
        assessments=assessments,
    )

    default_inputs = {"question": "What is MLflow?"}
    default_outputs = {"answer": "MLflow is a platform"}

    if _no_defaults:
        span_inputs = inputs
        span_outputs = outputs
    else:
        span_inputs = inputs if inputs is not None else default_inputs
        span_outputs = outputs if outputs is not None else default_outputs

    spans = [
        create_test_span(
            span_id=1,
            parent_id=None,
            name="root_span",
            inputs=span_inputs,
            outputs=span_outputs,
            span_type=SpanType.CHAIN,
        )
    ]

    trace_data = TraceData(spans=spans)
    return Trace(info=trace_info, data=trace_data)


def test_process_trace_records_with_dict_outputs():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(
        trace_id="trace1",
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is a platform", "confidence": 0.95},
    )

    record_dicts = dataset._process_trace_records([trace])

    assert len(record_dicts) == 1
    assert record_dicts[0]["inputs"] == {"question": "What is MLflow?"}
    assert record_dicts[0]["outputs"] == {"answer": "MLflow is a platform", "confidence": 0.95}
    assert record_dicts[0]["expectations"] == {}
    assert record_dicts[0]["source"]["source_type"] == DatasetRecordSourceType.TRACE.value
    assert record_dicts[0]["source"]["source_data"]["trace_id"] == "trace1"


def test_process_trace_records_with_string_outputs():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(
        trace_id="trace2",
        inputs={"query": "Tell me about Python"},
        outputs="Python is a programming language",
    )

    record_dicts = dataset._process_trace_records([trace])

    assert len(record_dicts) == 1
    assert record_dicts[0]["inputs"] == {"query": "Tell me about Python"}
    assert record_dicts[0]["outputs"] == "Python is a programming language"
    assert record_dicts[0]["expectations"] == {}
    assert record_dicts[0]["source"]["source_type"] == DatasetRecordSourceType.TRACE.value


def test_process_trace_records_with_non_dict_non_string_outputs():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(
        trace_id="trace3", inputs={"x": 1, "y": 2}, outputs=["result1", "result2", "result3"]
    )

    record_dicts = dataset._process_trace_records([trace])

    assert len(record_dicts) == 1
    assert record_dicts[0]["inputs"] == {"x": 1, "y": 2}
    assert record_dicts[0]["outputs"] == ["result1", "result2", "result3"]
    assert record_dicts[0]["source"]["source_type"] == DatasetRecordSourceType.TRACE.value


def test_process_trace_records_with_numeric_outputs():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(trace_id="trace4", inputs={"number": 42}, outputs=42)

    record_dicts = dataset._process_trace_records([trace])

    assert len(record_dicts) == 1
    assert record_dicts[0]["outputs"] == 42


def test_process_trace_records_with_none_outputs():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(
        trace_id="trace5", inputs={"input": "test"}, outputs=None, _no_defaults=True
    )

    record_dicts = dataset._process_trace_records([trace])

    assert len(record_dicts) == 1
    assert record_dicts[0]["outputs"] is None


def test_process_trace_records_with_expectations():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(
        trace_id="trace6",
        inputs={"question": "What is 2+2?"},
        outputs={"answer": "4"},
        expectations={"correctness": True, "tone": "neutral"},
    )

    record_dicts = dataset._process_trace_records([trace])

    assert len(record_dicts) == 1
    assert record_dicts[0]["expectations"] == {"correctness": True, "tone": "neutral"}


def test_process_trace_records_multiple_traces():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    traces = [
        create_test_trace(trace_id="trace1", outputs={"result": "answer1"}),
        create_test_trace(trace_id="trace2", outputs="string answer"),
        create_test_trace(trace_id="trace3", outputs=[1, 2, 3]),
    ]

    record_dicts = dataset._process_trace_records(traces)

    assert len(record_dicts) == 3
    assert record_dicts[0]["outputs"] == {"result": "answer1"}
    assert record_dicts[1]["outputs"] == "string answer"
    assert record_dicts[2]["outputs"] == [1, 2, 3]


def test_process_trace_records_mixed_types_error():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        digest="digest123",
        created_time=123456789,
        last_update_time=123456789,
    )

    trace = create_test_trace(trace_id="trace1")
    not_a_trace = {"not": "a trace"}

    with pytest.raises(
        MlflowException,
        match=(
            "Mixed types in trace list.*Expected all elements to be Trace objects.*"
            "element at index 1 is dict"
        ),
    ):
        dataset._process_trace_records([trace, not_a_trace])
