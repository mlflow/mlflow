import os
import sys

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import (
    create_evaluation_dataset,
    delete_evaluation_dataset,
    get_evaluation_dataset,
    search_evaluation_datasets,
)
from mlflow.tracking import MlflowClient


@pytest.fixture(params=["sqlalchemy"], autouse=True)
def tracking_uri(request, tmp_path):
    """Set an MLflow Tracking URI for all tests automatically."""
    if "MLFLOW_SKINNY" in os.environ:
        pytest.skip("SQLAlchemy store is not available in skinny.")

    original_tracking_uri = mlflow.get_tracking_uri()

    path = tmp_path.joinpath("mlflow.db").as_uri()
    tracking_uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[
        len("file://") :
    ]

    mlflow.set_tracking_uri(tracking_uri)

    yield tracking_uri

    mlflow.set_tracking_uri(original_tracking_uri)


@pytest.fixture
def client(tracking_uri):
    return MlflowClient(tracking_uri=tracking_uri)


@pytest.fixture
def experiments(client):
    exp1 = client.create_experiment("test_exp_1")
    exp2 = client.create_experiment("test_exp_2")
    exp3 = client.create_experiment("test_exp_3")
    return [exp1, exp2, exp3]


def test_create_and_get_evaluation_dataset(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="qa_evaluation_v1",
        experiment_ids=[experiments[0], experiments[1]],
        source_type="HUMAN",
        source="manual_curation",
    )

    assert dataset.name == "qa_evaluation_v1"
    assert dataset.source_type == "HUMAN"
    assert dataset.source == "manual_curation"
    assert len(dataset.experiment_ids) == 2
    assert dataset.dataset_id is not None

    retrieved = get_evaluation_dataset(dataset_id=dataset.dataset_id)

    assert retrieved.dataset_id == dataset.dataset_id
    assert retrieved.name == dataset.name
    assert retrieved.source_type == dataset.source_type
    assert retrieved.source == dataset.source
    assert set(retrieved.experiment_ids) == {experiments[0], experiments[1]}


def test_create_dataset_minimal_params(tracking_uri):
    dataset = create_evaluation_dataset(name="minimal_dataset")

    assert dataset.name == "minimal_dataset"
    assert dataset.source_type is None
    assert dataset.source is None
    assert dataset.experiment_ids == []


def test_active_record_pattern_merge_records(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="active_record_test",
        experiment_ids=experiments[0],
    )

    records_batch1 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an open source platform"},
            "tags": {"difficulty": "easy"},
        },
        {
            "inputs": {"question": "What is Python?"},
            "expectations": {"answer": "Python is a programming language"},
            "tags": {"difficulty": "easy"},
        },
    ]

    records_batch2 = [
        {
            "inputs": {"question": "What is MLflow?"},  # Duplicate input
            "expectations": {"answer": "MLflow is an ML lifecycle platform"},  # Updated expectation
            "tags": {"category": "ml"},  # Additional tag
        },
        {
            "inputs": {"question": "What is Docker?"},  # New record
            "expectations": {"answer": "Docker is a containerization platform"},
            "tags": {"difficulty": "medium"},
        },
    ]

    dataset.merge_records(records_batch1)

    df1 = dataset.to_df()
    assert len(df1) == 2

    mlflow_record = df1[df1["inputs"].apply(lambda x: x.get("question") == "What is MLflow?")].iloc[
        0
    ]
    assert mlflow_record["expectations"]["answer"] == "MLflow is an open source platform"
    assert mlflow_record["tags"]["difficulty"] == "easy"
    assert "category" not in mlflow_record["tags"]

    dataset.merge_records(records_batch2)

    df2 = dataset.to_df()
    assert len(df2) == 3  # Should have 3 unique records total

    mlflow_record_updated = df2[
        df2["inputs"].apply(lambda x: x.get("question") == "What is MLflow?")
    ].iloc[0]
    assert mlflow_record_updated["expectations"]["answer"] == "MLflow is an ML lifecycle platform"
    assert mlflow_record_updated["tags"]["difficulty"] == "easy"  # Original tag preserved
    assert mlflow_record_updated["tags"]["category"] == "ml"  # New tag added


def test_dataset_with_dataframe_records(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="dataframe_test",
        experiment_ids=experiments[0],
        source_type="CSV",
        source="test_data.csv",
    )

    df = pd.DataFrame(
        [
            {
                "inputs": {"text": "The movie was amazing!", "model": "sentiment-v1"},
                "expectations": {"sentiment": "positive", "confidence": 0.95},
                "tags": {"source": "imdb"},
            },
            {
                "inputs": {"text": "Terrible experience", "model": "sentiment-v1"},
                "expectations": {"sentiment": "negative", "confidence": 0.88},
                "tags": {"source": "yelp"},
            },
        ]
    )

    dataset.merge_records(df)

    result_df = dataset.to_df()
    assert len(result_df) == 2
    assert all(col in result_df.columns for col in ["inputs", "expectations", "tags"])

    first_record = result_df.iloc[0]
    assert first_record["inputs"]["text"] == "The movie was amazing!"
    assert first_record["expectations"]["sentiment"] == "positive"


def test_search_evaluation_datasets(tracking_uri, experiments):
    datasets = []
    for i in range(5):
        dataset = create_evaluation_dataset(
            name=f"search_test_{i}",
            experiment_ids=[experiments[i % len(experiments)]],
            source_type="HUMAN" if i % 2 == 0 else "TRACE",
        )
        datasets.append(dataset)

    all_results = search_evaluation_datasets()
    assert len(all_results) == 5

    exp0_results = search_evaluation_datasets(experiment_ids=experiments[0])
    assert len(exp0_results) == 2  # Datasets 0 and 3

    human_results = search_evaluation_datasets(filter_string="name LIKE 'search_test_%'")
    assert len(human_results) == 5

    page1 = search_evaluation_datasets(max_results=2)
    assert len(page1) == 2
    assert page1.token is not None

    page2 = search_evaluation_datasets(max_results=2, page_token=page1.token)
    assert len(page2) == 2


def test_delete_evaluation_dataset(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="to_be_deleted",
        experiment_ids=experiments[0],
    )

    dataset.merge_records([{"inputs": {"q": "test"}, "expectations": {"a": "answer"}}])

    retrieved = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    assert retrieved is not None

    delete_evaluation_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="not found"):
        get_evaluation_dataset(dataset_id=dataset.dataset_id)


def test_dataset_lifecycle_workflow(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="qa_eval_prod_v1",
        experiment_ids=[experiments[0], experiments[1]],
        source_type="HUMAN",
        source="qa_team_annotations",
    )

    initial_cases = [
        {
            "inputs": {"question": "What is the capital of France?"},
            "expectations": {"answer": "Paris", "confidence": "high"},
            "tags": {"category": "geography", "difficulty": "easy"},
        },
        {
            "inputs": {"question": "Explain quantum computing"},
            "expectations": {"answer": "Quantum computing uses quantum mechanics principles"},
            "tags": {"category": "science", "difficulty": "hard"},
        },
    ]
    dataset.merge_records(initial_cases)

    dataset_id = dataset.dataset_id
    retrieved = get_evaluation_dataset(dataset_id=dataset_id)
    df = retrieved.to_df()
    assert len(df) == 2

    additional_cases = [
        {
            "inputs": {"question": "What is 2+2?"},
            "expectations": {"answer": "4", "confidence": "high"},
            "tags": {"category": "math", "difficulty": "easy"},
        },
    ]
    retrieved.merge_records(additional_cases)

    found = search_evaluation_datasets(
        experiment_ids=experiments[0],
        filter_string="name LIKE 'qa_eval%'",
    )
    assert len(found) == 1
    assert found[0].dataset_id == dataset_id

    final_dataset = get_evaluation_dataset(dataset_id=dataset_id)
    final_df = final_dataset.to_df()
    assert len(final_df) == 3

    categories = set()
    for _, row in final_df.iterrows():
        if row["tags"] and "category" in row["tags"]:
            categories.add(row["tags"]["category"])
    assert categories == {"geography", "science", "math"}


def test_error_handling_filestore_backend(tmp_path):
    file_uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri(file_uri)

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        create_evaluation_dataset(name="test")
    assert exc.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        get_evaluation_dataset(dataset_id="test_id")
    assert exc.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        search_evaluation_datasets()
    assert exc.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        delete_evaluation_dataset(dataset_id="test_id")
    assert exc.value.error_code == "FEATURE_DISABLED"


def test_single_experiment_id_handling(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="single_exp_test",
        experiment_ids=experiments[0],  # Single string, not list
    )

    assert isinstance(dataset.experiment_ids, list)
    assert dataset.experiment_ids == [experiments[0]]

    results = search_evaluation_datasets(experiment_ids=experiments[0])
    found_ids = [d.dataset_id for d in results]
    assert dataset.dataset_id in found_ids


def test_trace_to_evaluation_dataset_integration(tracking_uri, experiments):
    trace_inputs = [
        {"question": "What is MLflow?", "context": "ML platforms"},
        {"question": "What is Python?", "context": "programming"},
        {"question": "What is MLflow?", "context": "ML platforms"},  # Duplicate
    ]

    created_trace_ids = []
    for i, inputs in enumerate(trace_inputs):
        with mlflow.start_run(experiment_id=experiments[i % 2]):
            with mlflow.start_span(name=f"qa_trace_{i}") as span:
                span.set_inputs(inputs)
                span.set_outputs({"answer": f"Answer for {inputs['question']}"})
                span.set_attributes({"model": "test-model", "temperature": "0.7"})
                trace_id = span.request_id
                created_trace_ids.append(trace_id)

    traces = mlflow.search_traces(
        experiment_ids=[experiments[0], experiments[1]],
        max_results=10,
        return_type="list",
    )
    assert len(traces) == 3

    dataset = create_evaluation_dataset(
        name="trace_eval_dataset",
        experiment_ids=[experiments[0], experiments[1]],
        source_type="TRACE",
        source="test_traces",
    )

    dataset.merge_records(traces)

    df = dataset.to_df()
    assert len(df) == 2  # Should have 2 unique inputs (3rd trace is duplicate)

    for _, record in df.iterrows():
        assert "inputs" in record
        assert "question" in record["inputs"]
        assert "context" in record["inputs"]
        assert record.get("source_type") == "TRACE"
        assert record.get("source_id") is not None  # Should have trace ID

    mlflow_records = df[df["inputs"].apply(lambda x: x.get("question") == "What is MLflow?")]
    assert len(mlflow_records) == 1

    with mlflow.start_run(experiment_id=experiments[0]):
        with mlflow.start_span(name="additional_trace") as span:
            span.set_inputs({"question": "What is Docker?", "context": "containers"})
            span.set_outputs({"answer": "Docker is a containerization platform"})
            span.set_attributes({"model": "test-model"})

    all_traces = mlflow.search_traces(
        experiment_ids=[experiments[0], experiments[1]], max_results=10, return_type="list"
    )
    assert len(all_traces) == 4

    new_trace = None
    for trace in all_traces:
        root_span = trace.data._get_root_span() if hasattr(trace, "data") else None
        if root_span and root_span.inputs and root_span.inputs.get("question") == "What is Docker?":
            new_trace = trace
            break

    assert new_trace is not None, "Could not find the Docker trace"

    dataset.merge_records([new_trace])

    final_df = dataset.to_df()
    assert len(final_df) == 3

    retrieved = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    retrieved_df = retrieved.to_df()
    assert len(retrieved_df) == 3

    delete_evaluation_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="not found"):
        get_evaluation_dataset(dataset_id=dataset.dataset_id)

    search_results = search_evaluation_datasets(
        experiment_ids=[experiments[0], experiments[1]], max_results=100
    )
    found_dataset_ids = [d.dataset_id for d in search_results]
    assert dataset.dataset_id not in found_dataset_ids

    all_datasets = search_evaluation_datasets(max_results=100)
    all_dataset_ids = [d.dataset_id for d in all_datasets]
    assert dataset.dataset_id not in all_dataset_ids
