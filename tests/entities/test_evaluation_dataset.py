import time
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType, Expectation
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.evaluation_dataset import EvaluationDataset
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.protos.evaluation_datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource
from mlflow.protos.evaluation_datasets_pb2 import EvaluationDataset as ProtoEvaluationDataset


@pytest.fixture(params=["file"], autouse=False)
def tracking_uri(request, tmp_path):
    original_tracking_uri = mlflow.get_tracking_uri()

    if request.param == "file":
        tracking_uri = tmp_path.joinpath("file").as_uri()

    mlflow.tracing.disable()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.tracing.enable()

    yield tracking_uri

    mlflow.set_tracking_uri(original_tracking_uri)


def test_evaluation_dataset_creation():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        source="manual",
        source_type="HUMAN",
        schema='{"fields": ["input", "output"]}',
        profile='{"count": 100}',
        digest="abc123",
        created_by="user1",
        last_updated_by="user2",
    )

    assert dataset.dataset_id == "dataset123"
    assert dataset.name == "test_dataset"
    assert dataset.source == "manual"
    assert dataset.source_type == "HUMAN"
    assert dataset.schema == '{"fields": ["input", "output"]}'
    assert dataset.profile == '{"count": 100}'
    assert dataset.digest == "abc123"
    assert dataset.created_by == "user1"
    assert dataset.last_updated_by == "user2"
    assert dataset.created_time is not None
    assert dataset.last_update_time is not None

    # Set experiment_ids after creation to test setter
    dataset.experiment_ids = ["exp1", "exp2"]
    assert dataset.experiment_ids == ["exp1", "exp2"]


def test_evaluation_dataset_auto_timestamps():
    dataset = EvaluationDataset(name="test_dataset")

    assert dataset.created_time is not None
    assert dataset.last_update_time is not None
    assert dataset.created_time > 0
    assert dataset.last_update_time > 0


def test_evaluation_dataset_experiment_ids_lazy_loading():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    assert dataset._experiment_ids is None

    mock_store = mock.Mock()
    mock_experiment_ids = ["exp1", "exp2", "exp3"]
    mock_store.get_evaluation_dataset_experiment_ids.return_value = mock_experiment_ids

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=mock_store):
        experiment_ids = dataset.experiment_ids
        assert experiment_ids == mock_experiment_ids
        mock_store.get_evaluation_dataset_experiment_ids.assert_called_once_with("dataset123")

        experiment_ids2 = dataset.experiment_ids
        assert experiment_ids2 == mock_experiment_ids
        assert mock_store.get_evaluation_dataset_experiment_ids.call_count == 1


def test_evaluation_dataset_experiment_ids_setter():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

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


def test_evaluation_dataset_proto_with_unloaded_experiment_ids():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    assert dataset._experiment_ids is None

    proto = dataset.to_proto()
    assert len(proto.experiment_ids) == 0
    assert dataset._experiment_ids is None


def test_evaluation_dataset_to_dict_triggers_lazy_loading():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    assert dataset._experiment_ids is None
    assert dataset._records is None

    mock_store = mock.Mock()
    mock_experiment_ids = ["exp1", "exp2"]
    mock_records = [
        DatasetRecord(
            dataset_id="dataset123",
            inputs={"question": "What is MLflow?"},
            expectations={"answer": "MLflow is..."},
        )
    ]
    mock_store.get_evaluation_dataset_experiment_ids.return_value = mock_experiment_ids
    mock_store._load_dataset_records.return_value = mock_records

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=mock_store):
        data = dataset.to_dict()

        assert data["experiment_ids"] == mock_experiment_ids
        assert len(data["records"]) == 1
        assert data["records"][0]["inputs"]["question"] == "What is MLflow?"

        mock_store.get_evaluation_dataset_experiment_ids.assert_called_once_with("dataset123")
        mock_store._load_dataset_records.assert_called_once_with("dataset123")


def test_evaluation_dataset_to_dict_no_backend_support():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    mock_store = mock.Mock()
    del mock_store.get_evaluation_dataset_experiment_ids
    del mock_store._load_dataset_records

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=mock_store):
        data = dataset.to_dict()

        assert data["experiment_ids"] == []
        assert data["records"] == []


def test_evaluation_dataset_lazy_loading():
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


def test_evaluation_dataset_merge_records_from_dict():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is a platform"},
            "tags": {"source": "manual"},
        },
        {
            "inputs": {"question": "How to use MLflow?"},
            "expectations": {"answer": "You can use MLflow by..."},
        },
    ]

    dataset.merge_records(records)

    assert len(dataset._records) == 2
    assert dataset._records[0].inputs["question"] == "What is MLflow?"
    assert dataset._records[0].expectations["answer"] == "MLflow is a platform"
    assert dataset._records[0].tags["source"] == "manual"
    assert dataset._records[1].inputs["question"] == "How to use MLflow?"


def test_evaluation_dataset_merge_records_from_dataframe():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    df = pd.DataFrame(
        [
            {
                "inputs": {"question": "What is MLflow?"},
                "expectations": {"answer": "MLflow is a platform"},
                "tags": {"source": "manual"},
            },
            {
                "inputs": {"question": "How to use MLflow?"},
                "expectations": {"answer": "You can use MLflow by..."},
            },
        ]
    )

    dataset.merge_records(df)

    assert len(dataset._records) == 2
    assert dataset._records[0].inputs["question"] == "What is MLflow?"
    assert dataset._records[1].inputs["question"] == "How to use MLflow?"


def test_evaluation_dataset_merge_records_from_traces(tracking_uri):
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    with mlflow.start_span(name="trace1") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is a platform"})
        trace_id1 = span.trace_id

    assessment_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="test_user"
    )

    expectation1 = Expectation(
        name="expected_answer", value="MLflow is an ML platform", source=assessment_source
    )
    mlflow.log_expectation(
        trace_id=trace_id1,
        name=expectation1.name,
        value=expectation1.value,
        source=expectation1.source,
    )

    expectation2 = Expectation(
        name="expected_facts", value=["fact1", "fact2"], source=assessment_source
    )
    mlflow.log_expectation(
        trace_id=trace_id1,
        name=expectation2.name,
        value=expectation2.value,
        source=expectation2.source,
    )

    with mlflow.start_span(name="trace2") as span:
        span.set_inputs({"question": "How to use MLflow?"})
        span.set_outputs({"answer": "You can use MLflow by..."})
        trace_id2 = span.trace_id

    expectation3 = Expectation(
        name="expected_answer", value="Use MLflow for ML lifecycle", source=assessment_source
    )
    mlflow.log_expectation(
        trace_id=trace_id2,
        name=expectation3.name,
        value=expectation3.value,
        source=expectation3.source,
    )

    with mlflow.start_span(name="trace3") as span:
        span.set_inputs({"question": "What is a model?"})
        span.set_outputs({"answer": "A model is..."})
        trace_id3 = span.trace_id

    trace1 = mlflow.get_trace(trace_id1)
    trace2 = mlflow.get_trace(trace_id2)
    trace3 = mlflow.get_trace(trace_id3)

    dataset.merge_records([trace1, trace2, trace3])

    assert len(dataset._records) == 3
    assert dataset._records[0].inputs["question"] == "What is MLflow?"
    assert dataset._records[0].expectations["expected_answer"] == "MLflow is an ML platform"
    assert dataset._records[0].expectations["expected_facts"] == ["fact1", "fact2"]
    assert dataset._records[0].source.source_type == "TRACE"
    assert dataset._records[0].source.source_data["trace_id"] == trace_id1

    assert dataset._records[1].inputs["question"] == "How to use MLflow?"
    assert dataset._records[1].expectations["expected_answer"] == "Use MLflow for ML lifecycle"
    assert dataset._records[1].source.source_data["trace_id"] == trace_id2

    assert dataset._records[2].inputs["question"] == "What is a model?"
    assert dataset._records[2].expectations == {}
    assert dataset._records[2].source.source_data["trace_id"] == trace_id3


def test_evaluation_dataset_merge_records_validation():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    with pytest.raises(MlflowException, match="Each record must be a dictionary"):
        dataset.merge_records(["invalid", "records"])

    with pytest.raises(MlflowException, match="Each record must have an 'inputs' field"):
        dataset.merge_records([{"expectations": {"answer": "test"}}])


def test_evaluation_dataset_merge_records_with_tracking_store():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    mock_store = mock.Mock()
    mock_store.get_evaluation_dataset.return_value = dataset  # Mock successful dataset retrieval
    mock_store.upsert_evaluation_dataset_records.return_value = {"inserted": 2, "updated": 0}

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=mock_store):
        records = [{"inputs": {"question": "What is MLflow?"}}]

        result = dataset.merge_records(records)

        assert result is dataset

        mock_store.get_evaluation_dataset.assert_called_once_with("dataset123")
        mock_store.upsert_evaluation_dataset_records.assert_called_once_with(
            dataset_id="dataset123", records=records, updated_by=None
        )

        assert dataset._records is None


def test_evaluation_dataset_to_df():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    df = dataset.to_df()
    assert len(df) == 0
    assert list(df.columns) == ["inputs", "expectations", "tags", "source_type", "source_id"]

    dataset._records = [
        DatasetRecord(
            dataset_record_id="rec1",
            dataset_id="dataset123",
            inputs={"question": "What is MLflow?"},
            expectations={"answer": "MLflow is a platform"},
            tags={"source": "manual"},
            source={"source_type": "HUMAN", "source_data": {"user_id": "user1"}},
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


def test_evaluation_dataset_to_from_proto():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        source="manual",
        source_type="HUMAN",
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
    assert isinstance(proto, ProtoEvaluationDataset)
    assert proto.dataset_id == "dataset123"
    assert proto.name == "test_dataset"
    assert proto.source == "manual"
    assert proto.source_type == ProtoDatasetRecordSource.SourceType.Value("HUMAN")
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
    assert dataset2.source == dataset.source
    assert dataset2.source_type == dataset.source_type
    assert dataset2.schema == dataset.schema
    assert dataset2.profile == dataset.profile
    assert dataset2.digest == dataset.digest
    assert dataset2.created_time == dataset.created_time
    assert dataset2.last_update_time == dataset.last_update_time
    assert dataset2.created_by == dataset.created_by
    assert dataset2.last_updated_by == dataset.last_updated_by
    assert dataset2._experiment_ids == ["exp1", "exp2"]
    assert dataset2.experiment_ids == ["exp1", "exp2"]


def test_evaluation_dataset_to_from_dict():
    dataset = EvaluationDataset(
        dataset_id="dataset123",
        name="test_dataset",
        source="manual",
        source_type="HUMAN",
    )
    dataset.experiment_ids = ["exp1", "exp2"]

    dataset._records = [
        DatasetRecord(dataset_id="dataset123", inputs={"question": "What is MLflow?"})
    ]

    data = dataset.to_dict()
    assert data["dataset_id"] == "dataset123"
    assert data["name"] == "test_dataset"
    assert data["source"] == "manual"
    assert data["source_type"] == "HUMAN"
    assert data["experiment_ids"] == ["exp1", "exp2"]
    assert len(data["records"]) == 1
    assert data["records"][0]["inputs"]["question"] == "What is MLflow?"

    dataset2 = EvaluationDataset.from_dict(data)
    assert dataset2.dataset_id == dataset.dataset_id
    assert dataset2.name == dataset.name
    assert dataset2.source == dataset.source
    assert dataset2.source_type == dataset.source_type
    assert dataset2._experiment_ids == ["exp1", "exp2"]
    assert dataset2.experiment_ids == ["exp1", "exp2"]
    assert len(dataset2._records) == 1
    assert dataset2._records[0].inputs["question"] == "What is MLflow?"


def test_evaluation_dataset_merge_records_from_traces_edge_cases():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    mock_expectation1 = mock.Mock()
    mock_expectation1.name = "expected_response"
    mock_expectation1.value = "Expected value"

    mock_trace1 = mock.Mock(spec=Trace)
    mock_trace1.info = mock.Mock()
    mock_trace1.info.trace_id = "trace1"
    mock_trace1.data = mock.Mock()
    mock_trace1.data.spans = []
    mock_trace1.data._get_root_span = mock.Mock(return_value=None)
    mock_trace1.search_assessments = mock.Mock(return_value=[mock_expectation1])

    mock_span = mock.Mock()
    mock_span.parent_id = "parent123"
    mock_span.inputs = {"question": "Should not be used"}
    mock_span.outputs = {"answer": "Should not be used"}

    mock_trace2 = mock.Mock(spec=Trace)
    mock_trace2.info = mock.Mock()
    mock_trace2.info.trace_id = "trace2"
    mock_trace2.data = mock.Mock()
    mock_trace2.data.spans = [mock_span]
    mock_trace2.data._get_root_span = mock.Mock(return_value=None)
    mock_trace2.search_assessments = mock.Mock(return_value=[])

    mock_span3 = mock.Mock()
    mock_span3.parent_id = None
    mock_span3.inputs = None
    mock_span3.outputs = None

    mock_trace3 = mock.Mock(spec=Trace)
    mock_trace3.info = mock.Mock()
    mock_trace3.info.trace_id = "trace3"
    mock_trace3.data = mock.Mock()
    mock_trace3.data.spans = [mock_span3]
    mock_trace3.data._get_root_span = mock.Mock(return_value=mock_span3)
    mock_trace3.search_assessments = mock.Mock(return_value=[])

    mock_expectation2 = mock.Mock()
    mock_expectation2.name = "direct_value"
    mock_expectation2.value = "Direct expectation value"

    mock_trace4 = mock.Mock(spec=Trace)
    mock_trace4.info = mock.Mock()
    mock_trace4.info.trace_id = "trace4"
    mock_trace4.data = mock.Mock()
    mock_trace4.data.spans = []
    mock_trace4.data._get_root_span = mock.Mock(return_value=None)
    mock_trace4.search_assessments = mock.Mock(return_value=[mock_expectation2])

    dataset.merge_records([mock_trace1, mock_trace2, mock_trace3, mock_trace4])

    assert len(dataset._records) == 1

    record = dataset._records[0]
    assert record.inputs == {}

    assert record.expectations == {
        "expected_response": "Expected value",  # From trace1
        "direct_value": "Direct expectation value",  # From trace4
    }

    assert record.source.source_data["trace_id"] == "trace1"


def test_evaluation_dataset_method_chaining():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    result = dataset.merge_records([{"inputs": {"q1": "test1"}}]).merge_records(
        [{"inputs": {"q2": "test2"}}]
    )

    assert result is dataset
    assert len(dataset._records) == 2


def test_evaluation_dataset_merge_records_duplicate_inputs():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset.merge_records(
        [
            {
                "inputs": {"question": "What is MLflow?"},
                "expectations": {"answer": "MLflow is a platform"},
                "tags": {"source": "manual", "quality": "high"},
            }
        ]
    )

    dataset.merge_records(
        [
            {
                "inputs": {"question": "What is MLflow?"},
                "expectations": {"answer": "MLflow is an ML platform", "score": 0.9},
                "tags": {"source": "automated", "version": "v2"},
            }
        ]
    )

    assert len(dataset._records) == 1

    record = dataset._records[0]

    assert record.inputs == {"question": "What is MLflow?"}

    assert record.expectations == {"answer": "MLflow is an ML platform", "score": 0.9}

    assert record.tags == {"source": "automated", "quality": "high", "version": "v2"}


def test_evaluation_dataset_merge_records_with_partial_duplicates():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset.merge_records(
        [
            {
                "inputs": {"question": "What is Spark?"},
                "expectations": {"answer": "Spark is a data processing engine"},
                "tags": {"category": "big_data", "difficulty": "medium"},
            },
            {
                "inputs": {"question": "What is MLflow?"},
                "expectations": {"answer": "MLflow is a platform"},
                "tags": {"category": "ml_ops"},
            },
        ]
    )

    dataset.merge_records(
        [
            {
                "inputs": {"question": "What is Spark?"},
                "expectations": {
                    "answer": "Apache Spark is a unified analytics engine",
                    "confidence": 0.95,
                },
                "tags": {"category": "apache", "difficulty": "medium", "version": "3.0"},
            },
            {
                "inputs": {"question": "What is Python?"},
                "expectations": {"answer": "Python is a programming language"},
                "tags": {"category": "programming"},
            },
        ]
    )

    assert len(dataset._records) == 3

    records_by_question = {}
    for record in dataset._records:
        question = record.inputs.get("question")
        records_by_question[question] = record

    spark_record = records_by_question["What is Spark?"]
    assert spark_record.expectations == {
        "answer": "Apache Spark is a unified analytics engine",
        "confidence": 0.95,
    }
    assert spark_record.tags == {"category": "apache", "difficulty": "medium", "version": "3.0"}

    mlflow_record = records_by_question["What is MLflow?"]
    assert mlflow_record.expectations == {"answer": "MLflow is a platform"}
    assert mlflow_record.tags == {"category": "ml_ops"}

    python_record = records_by_question["What is Python?"]
    assert python_record.expectations == {"answer": "Python is a programming language"}
    assert python_record.tags == {"category": "programming"}


def test_evaluation_dataset_merge_records_empty_expectations_and_tags():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset.merge_records(
        [
            {
                "inputs": {"question": "What is MLflow?"},
                "expectations": {"answer": "MLflow is a platform"},
                "tags": {"source": "manual"},
            },
            {"inputs": {"question": "What is MLflow?"}, "tags": {"reviewed": "true"}},
            {"inputs": {"question": "What is MLflow?"}, "expectations": {"score": 0.8}},
        ]
    )

    assert len(dataset._records) == 1

    record = dataset._records[0]
    assert record.expectations == {"answer": "MLflow is a platform", "score": 0.8}
    assert record.tags == {"source": "manual", "reviewed": "true"}


def test_evaluation_dataset_merge_records_from_traces_with_duplicates(tracking_uri):
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    traces = []

    with mlflow.start_span(name="trace1") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is a platform"})
        trace_id1 = span.trace_id

    assessment_source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1")

    expectation1 = Expectation(
        name="expected_answer", value="MLflow is an ML platform", source=assessment_source
    )
    mlflow.log_expectation(
        trace_id=trace_id1,
        name=expectation1.name,
        value=expectation1.value,
        source=expectation1.source,
    )
    traces.append(mlflow.get_trace(trace_id1))

    with mlflow.start_span(name="trace2") as span:
        span.set_inputs({"question": "What is MLflow?"})  # Same input
        span.set_outputs({"answer": "MLflow helps with ML lifecycle"})
        trace_id2 = span.trace_id

    expectation2 = Expectation(
        name="expected_answer", value="MLflow is an open source platform", source=assessment_source
    )
    expectation3 = Expectation(name="quality_score", value=0.9, source=assessment_source)
    mlflow.log_expectation(
        trace_id=trace_id2,
        name=expectation2.name,
        value=expectation2.value,
        source=expectation2.source,
    )
    mlflow.log_expectation(
        trace_id=trace_id2,
        name=expectation3.name,
        value=expectation3.value,
        source=expectation3.source,
    )
    traces.append(mlflow.get_trace(trace_id2))

    with mlflow.start_span(name="trace3") as span:
        span.set_inputs({"question": "What is Spark?"})
        span.set_outputs({"answer": "Spark is a data processing engine"})
        trace_id3 = span.trace_id

    expectation4 = Expectation(
        name="expected_answer",
        value="Apache Spark is a unified analytics engine",
        source=assessment_source,
    )
    mlflow.log_expectation(
        trace_id=trace_id3,
        name=expectation4.name,
        value=expectation4.value,
        source=expectation4.source,
    )
    traces.append(mlflow.get_trace(trace_id3))

    dataset.merge_records(traces)

    assert len(dataset._records) == 2

    records_by_question = {}
    for record in dataset._records:
        question = record.inputs.get("question")
        records_by_question[question] = record

    mlflow_record = records_by_question["What is MLflow?"]
    assert mlflow_record.expectations == {
        "expected_answer": "MLflow is an open source platform",
        "quality_score": 0.9,
    }

    assert mlflow_record.source.source_type == "TRACE"
    assert mlflow_record.source.source_data["trace_id"] == trace_id1
    assert mlflow_record.source_id == trace_id1

    spark_record = records_by_question["What is Spark?"]
    assert spark_record.expectations == {
        "expected_answer": "Apache Spark is a unified analytics engine"
    }


def test_evaluation_dataset_merge_traces_preserves_first_source():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    first_record = {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"answer": "MLflow is a platform"},
        "tags": {"version": "v1", "quality": "high"},
        "source": {
            "source_type": "TRACE",
            "source_data": {"trace_id": "trace-001", "span_id": "span-001"},
        },
    }
    dataset.merge_records([first_record])

    assert len(dataset._records) == 1
    assert dataset._records[0].source.source_type == "TRACE"
    assert dataset._records[0].source.source_data["trace_id"] == "trace-001"
    assert dataset._records[0].source.source_data["span_id"] == "span-001"
    assert dataset._records[0].source_id == "trace-001"
    assert dataset._records[0].source_type == "TRACE"

    second_record = {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"answer": "MLflow is an ML platform", "confidence": 0.9},
        "tags": {"version": "v2", "reviewed": "true"},
        "source": {
            "source_type": "TRACE",
            "source_data": {"trace_id": "trace-002", "span_id": "span-002"},
        },
    }
    dataset.merge_records([second_record])

    assert len(dataset._records) == 1
    record = dataset._records[0]

    assert record.expectations == {
        "answer": "MLflow is an ML platform",
        "confidence": 0.9,
    }
    assert record.tags == {
        "version": "v2",
        "quality": "high",
        "reviewed": "true",
    }

    assert record.source.source_type == "TRACE"
    assert record.source.source_data["trace_id"] == "trace-001"
    assert record.source.source_data["span_id"] == "span-001"
    assert record.source_id == "trace-001"
    assert record.source_type == "TRACE"


def test_evaluation_dataset_merge_updates_last_update_time():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    first_record = {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"answer": "MLflow is a platform"},
        "tags": {"version": "v1"},
    }
    dataset.merge_records([first_record])

    initial_update_time = dataset._records[0].last_update_time
    assert initial_update_time is not None
    assert initial_update_time > 0

    time.sleep(0.01)

    second_record = {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"answer": "MLflow is an ML platform", "score": 0.9},
        "tags": {"version": "v2", "reviewed": "true"},
    }
    dataset.merge_records([second_record])

    assert len(dataset._records) == 1
    record = dataset._records[0]

    assert record.last_update_time > initial_update_time

    assert record.created_time == dataset._records[0].created_time

    assert record.expectations == {"answer": "MLflow is an ML platform", "score": 0.9}
    assert record.tags == {"version": "v2", "reviewed": "true"}


def test_evaluation_dataset_merge_records_mixed_trace_types():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    mock_trace = mock.Mock(spec=Trace)
    mock_trace.info = mock.Mock()
    mock_trace.info.trace_id = "trace1"
    mock_trace.data = mock.Mock()
    mock_trace.data._get_root_span = mock.Mock(return_value=None)
    mock_trace.search_assessments = mock.Mock(return_value=[])

    mixed_records = [
        mock_trace,
        {"inputs": {"question": "What is Python?"}, "expectations": {"answer": "A language"}},
        mock_trace,
    ]

    with pytest.raises(
        MlflowException, match="Mixed types in trace list.*element at index 1 is dict"
    ):
        dataset.merge_records(mixed_records)

    mixed_records2 = [mock_trace, mock_trace, "not a trace"]

    with pytest.raises(
        MlflowException, match="Mixed types in trace list.*element at index 2 is str"
    ):
        dataset.merge_records(mixed_records2)


def test_evaluation_dataset_merge_records_empty_list():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset.merge_records([])
    assert len(dataset._records) == 0

    dataset.merge_records([{"inputs": {"q": "test"}, "expectations": {"a": "answer"}}])
    assert len(dataset._records) == 1

    dataset.merge_records([])
    assert len(dataset._records) == 1


def test_evaluation_dataset_merge_records_nonexistent_dataset():
    """Test that merge_records validates dataset exists in the tracking store."""
    dataset = EvaluationDataset(dataset_id="nonexistent123", name="test_dataset")

    # Create a mock store that has the methods but the dataset doesn't exist
    mock_store = mock.Mock()
    mock_store.get_evaluation_dataset.side_effect = Exception("Dataset not found")
    mock_store.upsert_evaluation_dataset_records = mock.Mock()

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=mock_store):
        records = [{"inputs": {"question": "What is MLflow?"}}]

        with pytest.raises(
            MlflowException, match="Cannot add records to dataset nonexistent123: Dataset not found"
        ):
            dataset.merge_records(records)

        # Verify we tried to check if dataset exists but didn't try to upsert
        mock_store.get_evaluation_dataset.assert_called_once_with("nonexistent123")
        mock_store.upsert_evaluation_dataset_records.assert_not_called()


def test_evaluation_dataset_merge_records_single_call_deduplication():
    dataset = EvaluationDataset(dataset_id="dataset123", name="test_dataset")

    dataset.merge_records(
        [
            {
                "inputs": {"question": "What is MLflow?", "context": "ML platforms"},
                "expectations": {"answer": "MLflow is a platform", "confidence": 0.8},
                "tags": {"source": "manual", "version": "v1"},
            },
            {
                "inputs": {"question": "What is Python?"},
                "expectations": {"answer": "Python is a programming language"},
                "tags": {"category": "programming"},
            },
            {
                "inputs": {"question": "What is MLflow?", "context": "ML platforms"},
                "expectations": {"answer": "MLflow is an ML lifecycle platform", "quality": "high"},
                "tags": {"source": "automated", "reviewed": "true"},
            },
        ]
    )

    assert len(dataset._records) == 2

    records_by_question = {}
    for record in dataset._records:
        question = record.inputs.get("question")
        records_by_question[question] = record

    mlflow_record = records_by_question["What is MLflow?"]
    assert mlflow_record.inputs == {"question": "What is MLflow?", "context": "ML platforms"}
    assert mlflow_record.expectations == {
        "answer": "MLflow is an ML lifecycle platform",
        "confidence": 0.8,
        "quality": "high",
    }
    assert mlflow_record.tags == {"source": "automated", "version": "v1", "reviewed": "true"}

    python_record = records_by_question["What is Python?"]
    assert python_record.expectations == {"answer": "Python is a programming language"}
    assert python_record.tags == {"category": "programming"}
