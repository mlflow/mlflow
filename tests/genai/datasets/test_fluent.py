import json
import os
import sys
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.entities.evaluation_dataset import EvaluationDataset as EntityEvaluationDataset
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import (
    EvaluationDataset,
    create_evaluation_dataset,
    delete_evaluation_dataset,
    delete_evaluation_dataset_tag,
    get_evaluation_dataset,
    search_evaluation_datasets,
    set_evaluation_dataset_tags,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_USER


@pytest.fixture
def mock_client():
    with mock.patch("mlflow.genai.datasets.MlflowClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        yield mock_client_instance


@pytest.fixture
def mock_databricks_environment():
    with mock.patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
        yield


@pytest.fixture(params=["sqlalchemy"], autouse=True)
def tracking_uri(request, tmp_path):
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


@pytest.fixture
def experiment(client):
    return client.create_experiment("test_trace_experiment")


def test_create_evaluation_dataset(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
        tags={"environment": "production", "version": "1.0"},
    )
    mock_client.create_evaluation_dataset.return_value = expected_dataset

    result = create_evaluation_dataset(
        name="test_dataset",
        experiment_ids=["exp1", "exp2"],
        tags={"environment": "production", "version": "1.0"},
    )

    assert result == expected_dataset
    mock_client.create_evaluation_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_ids=["exp1", "exp2"],
        tags={"environment": "production", "version": "1.0"},
    )


def test_create_evaluation_dataset_single_experiment_id(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
    )
    mock_client.create_evaluation_dataset.return_value = expected_dataset

    result = create_evaluation_dataset(
        name="test_dataset",
        experiment_ids="exp1",
    )

    assert result == expected_dataset
    mock_client.create_evaluation_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_ids=["exp1"],
        tags=None,
    )


def test_create_evaluation_dataset_with_empty_tags(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
        tags={},
    )
    mock_client.create_evaluation_dataset.return_value = expected_dataset

    result = create_evaluation_dataset(
        name="test_dataset",
        experiment_ids=["exp1"],
        tags={},
    )

    assert result == expected_dataset
    mock_client.create_evaluation_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_ids=["exp1"],
        tags={},
    )


def test_create_evaluation_dataset_databricks(mock_databricks_environment):
    mock_dataset = mock.Mock()
    with mock.patch.dict(
        "sys.modules",
        {
            "databricks.agents.datasets": mock.Mock(
                create_dataset=mock.Mock(return_value=mock_dataset)
            )
        },
    ):
        result = create_evaluation_dataset(
            name="catalog.schema.table",
            experiment_ids=["exp1", "exp2"],
        )

        sys.modules["databricks.agents.datasets"].create_dataset.assert_called_once_with(
            "catalog.schema.table", ["exp1", "exp2"]
        )
        assert isinstance(result, EvaluationDataset)


def test_get_evaluation_dataset(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
    )
    mock_client.get_evaluation_dataset.return_value = expected_dataset

    result = get_evaluation_dataset(dataset_id="test_id")

    assert result == expected_dataset
    mock_client.get_evaluation_dataset.assert_called_once_with("test_id")


def test_get_evaluation_dataset_missing_id():
    with pytest.raises(ValueError, match="Parameter 'dataset_id' is required"):
        get_evaluation_dataset()


def test_get_evaluation_dataset_databricks(mock_databricks_environment):
    mock_dataset = mock.Mock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.agents.datasets": mock.Mock(get_dataset=mock.Mock(return_value=mock_dataset))},
    ):
        result = get_evaluation_dataset(name="catalog.schema.table")

        sys.modules["databricks.agents.datasets"].get_dataset.assert_called_once_with(
            "catalog.schema.table"
        )
        assert isinstance(result, EvaluationDataset)


def test_get_evaluation_dataset_databricks_missing_name(mock_databricks_environment):
    with pytest.raises(ValueError, match="Parameter 'name' is required in Databricks environment"):
        get_evaluation_dataset(dataset_id="test_id")


def test_delete_evaluation_dataset(mock_client):
    delete_evaluation_dataset(dataset_id="test_id")

    mock_client.delete_evaluation_dataset.assert_called_once_with("test_id")


def test_delete_evaluation_dataset_missing_id():
    with pytest.raises(ValueError, match="Parameter 'dataset_id' is required"):
        delete_evaluation_dataset()


def test_delete_evaluation_dataset_databricks(mock_databricks_environment):
    with mock.patch.dict(
        "sys.modules", {"databricks.agents.datasets": mock.Mock(delete_dataset=mock.Mock())}
    ):
        delete_evaluation_dataset(name="catalog.schema.table")

        sys.modules["databricks.agents.datasets"].delete_dataset.assert_called_once_with(
            "catalog.schema.table"
        )


def test_search_evaluation_datasets(mock_client):
    datasets = [
        EntityEvaluationDataset(
            dataset_id="id1",
            name="dataset1",
            digest="digest1",
            created_time=123456789,
            last_update_time=123456789,
        ),
        EntityEvaluationDataset(
            dataset_id="id2",
            name="dataset2",
            digest="digest2",
            created_time=123456789,
            last_update_time=123456789,
        ),
    ]
    mock_client.search_evaluation_datasets.return_value = PagedList(datasets, "next_token")

    result = search_evaluation_datasets(
        experiment_ids=["exp1", "exp2"],
        filter_string="name LIKE 'test%'",
        max_results=100,
        order_by=["created_time DESC"],
        page_token="token123",
    )

    assert len(result) == 2
    assert result.token == "next_token"
    mock_client.search_evaluation_datasets.assert_called_once_with(
        experiment_ids=["exp1", "exp2"],
        filter_string="name LIKE 'test%'",
        max_results=100,
        order_by=["created_time DESC"],
        page_token="token123",
    )


def test_search_evaluation_datasets_single_experiment_id(mock_client):
    datasets = [
        EntityEvaluationDataset(
            dataset_id="id1",
            name="dataset1",
            digest="digest1",
            created_time=123456789,
            last_update_time=123456789,
        )
    ]
    mock_client.search_evaluation_datasets.return_value = PagedList(datasets, None)

    search_evaluation_datasets(experiment_ids="exp1")

    mock_client.search_evaluation_datasets.assert_called_once_with(
        experiment_ids=["exp1"],
        filter_string=None,
        max_results=SEARCH_EVALUATION_DATASETS_MAX_RESULTS,
        order_by=None,
        page_token=None,
    )


def test_search_evaluation_datasets_databricks(mock_databricks_environment):
    with pytest.raises(
        NotImplementedError, match="Evaluation Dataset search is not available in Databricks"
    ):
        search_evaluation_datasets()


def test_databricks_import_error():
    with mock.patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
        with mock.patch.dict("sys.modules", {"databricks.agents.datasets": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="databricks-agents"):
                    create_evaluation_dataset(name="test", experiment_ids="exp1")


def test_create_evaluation_dataset_with_user_tag(tracking_uri, experiments):
    """Test that mlflow.user tag properly sets the created_by field."""
    dataset = create_evaluation_dataset(
        name="test_user_attribution",
        experiment_ids=experiments[0],
        tags={"environment": "test", MLFLOW_USER: "john_doe"},
    )

    assert dataset.name == "test_user_attribution"
    assert dataset.tags[MLFLOW_USER] == "john_doe"
    assert dataset.created_by == "john_doe"

    dataset2 = create_evaluation_dataset(
        name="test_no_user",
        experiment_ids=experiments[0],
        tags={"environment": "test"},
    )

    assert dataset2.name == "test_no_user"
    assert isinstance(dataset2.tags[MLFLOW_USER], str)
    assert dataset2.created_by == dataset2.tags[MLFLOW_USER]


def test_create_and_get_evaluation_dataset(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="qa_evaluation_v1",
        experiment_ids=[experiments[0], experiments[1]],
        tags={"source": "manual_curation", "environment": "test"},
    )

    assert dataset.name == "qa_evaluation_v1"
    assert dataset.tags["source"] == "manual_curation"
    assert dataset.tags["environment"] == "test"
    assert len(dataset.experiment_ids) == 2
    assert dataset.dataset_id is not None

    retrieved = get_evaluation_dataset(dataset_id=dataset.dataset_id)

    assert retrieved.dataset_id == dataset.dataset_id
    assert retrieved.name == dataset.name
    assert retrieved.tags == dataset.tags
    assert set(retrieved.experiment_ids) == {experiments[0], experiments[1]}


def test_create_dataset_minimal_params(tracking_uri):
    dataset = create_evaluation_dataset(name="minimal_dataset")

    assert dataset.name == "minimal_dataset"
    assert "mlflow.user" not in dataset.tags or isinstance(dataset.tags.get("mlflow.user"), str)
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
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML lifecycle platform"},
            "tags": {"category": "ml"},
        },
        {
            "inputs": {"question": "What is Docker?"},
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
    assert len(df2) == 3

    mlflow_record_updated = df2[
        df2["inputs"].apply(lambda x: x.get("question") == "What is MLflow?")
    ].iloc[0]
    assert mlflow_record_updated["expectations"]["answer"] == "MLflow is an ML lifecycle platform"
    assert mlflow_record_updated["tags"]["difficulty"] == "easy"
    assert mlflow_record_updated["tags"]["category"] == "ml"


def test_dataset_with_dataframe_records(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="dataframe_test",
        experiment_ids=experiments[0],
        tags={"source": "csv", "file": "test_data.csv"},
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
            tags={"type": "human" if i % 2 == 0 else "trace", "index": str(i)},
        )
        datasets.append(dataset)

    all_results = search_evaluation_datasets()
    assert len(all_results) == 5

    exp0_results = search_evaluation_datasets(experiment_ids=experiments[0])
    assert len(exp0_results) == 2

    human_results = search_evaluation_datasets(filter_string="name LIKE 'search_test_%'")
    assert len(human_results) == 5

    human_type_results = search_evaluation_datasets(filter_string="tags.type = 'human'")
    assert len(human_type_results) == 3
    assert all(ds.tags.get("type") == "human" for ds in human_type_results)

    combined_results = search_evaluation_datasets(
        experiment_ids=experiments[0], filter_string="tags.type = 'trace'"
    )
    assert len(combined_results) == 1

    ordered_results = search_evaluation_datasets(
        filter_string="tags.type = 'trace'", order_by=["name DESC"]
    )
    assert len(ordered_results) == 2
    names = [ds.name for ds in ordered_results]
    assert names == sorted(names, reverse=True)

    index_results = search_evaluation_datasets(filter_string="tags.index != '0'")
    assert len(index_results) == 4
    assert all(ds.tags.get("index") != "0" for ds in index_results)

    page1 = search_evaluation_datasets(max_results=2)
    assert len(page1) == 2
    assert page1.token is not None

    page2 = search_evaluation_datasets(max_results=2, page_token=page1.token)
    assert len(page2) == 2


def test_delete_evaluation_dataset(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="to_be_deleted",
        experiment_ids=[experiments[0], experiments[1]],
        tags={"env": "test", "version": "1.0"},
    )
    dataset_id = dataset.dataset_id

    dataset.merge_records([{"inputs": {"q": "test"}, "expectations": {"a": "answer"}}])

    retrieved = get_evaluation_dataset(dataset_id=dataset_id)
    assert retrieved is not None
    assert len(retrieved.to_df()) == 1

    delete_evaluation_dataset(dataset_id=dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        get_evaluation_dataset(dataset_id=dataset_id)

    search_results = search_evaluation_datasets(experiment_ids=[experiments[0], experiments[1]])
    found_ids = [d.dataset_id for d in search_results]
    assert dataset_id not in found_ids


def test_dataset_lifecycle_workflow(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="qa_eval_prod_v1",
        experiment_ids=[experiments[0], experiments[1]],
        tags={"source": "qa_team_annotations", "team": "qa", "env": "prod"},
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
        experiment_ids=experiments[0],
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
        {"question": "What is MLflow?", "context": "ML platforms"},
    ]

    created_trace_ids = []
    for i, inputs in enumerate(trace_inputs):
        with mlflow.start_run(experiment_id=experiments[i % 2]):
            with mlflow.start_span(name=f"qa_trace_{i}") as span:
                span.set_inputs(inputs)
                span.set_outputs({"answer": f"Answer for {inputs['question']}"})
                span.set_attributes({"model": "test-model", "temperature": "0.7"})
                trace_id = span.trace_id
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
        tags={"source": "test_traces", "type": "trace_integration"},
    )

    dataset.merge_records(traces)

    df = dataset.to_df()
    assert len(df) == 2

    for _, record in df.iterrows():
        assert "inputs" in record
        assert "question" in record["inputs"]
        assert "context" in record["inputs"]
        assert record.get("source_type") == "TRACE"
        assert record.get("source_id") is not None

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

    assert new_trace is not None

    dataset.merge_records([new_trace])

    final_df = dataset.to_df()
    assert len(final_df) == 3

    retrieved = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    retrieved_df = retrieved.to_df()
    assert len(retrieved_df) == 3

    delete_evaluation_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        get_evaluation_dataset(dataset_id=dataset.dataset_id)

    search_results = search_evaluation_datasets(
        experiment_ids=[experiments[0], experiments[1]], max_results=100
    )
    found_dataset_ids = [d.dataset_id for d in search_results]
    assert dataset.dataset_id not in found_dataset_ids

    all_datasets = search_evaluation_datasets(max_results=100)
    all_dataset_ids = [d.dataset_id for d in all_datasets]
    assert dataset.dataset_id not in all_dataset_ids


def test_trace_to_dataset_with_assessments(client, experiment):
    trace_data = [
        {
            "inputs": {"question": "What is MLflow?", "context": "ML platforms"},
            "outputs": {"answer": "MLflow is an open source platform for ML lifecycle"},
            "expectations": {
                "correctness": True,
                "completeness": 0.8,
            },
        },
        {
            "inputs": {"question": "What is Python?", "context": "programming languages"},
            "outputs": {"answer": "Python is a high-level programming language"},
            "expectations": {
                "correctness": True,
            },
        },
        {
            "inputs": {"question": "What is Docker?", "context": "containerization"},
            "outputs": {"answer": "Docker is a container platform"},
            "expectations": {},
        },
    ]

    created_traces = []
    for i, data in enumerate(trace_data):
        with mlflow.start_run(experiment_id=experiment):
            with mlflow.start_span(name=f"qa_trace_{i}") as span:
                span.set_inputs(data["inputs"])
                span.set_outputs(data["outputs"])
                span.set_attributes({"model": "test-model", "temperature": 0.7})
                trace_id = span.trace_id

                for name, value in data["expectations"].items():
                    mlflow.log_expectation(
                        trace_id=trace_id,
                        name=name,
                        value=value,
                        span_id=span.span_id,
                    )

        trace = client.get_trace(trace_id)
        created_traces.append(trace)

    dataset = create_evaluation_dataset(
        name="trace_assessment_dataset",
        experiment_ids=[experiment],
        tags={"source": "trace_integration_test", "version": "1.0"},
    )

    dataset.merge_records(created_traces)

    df = dataset.to_df()
    assert len(df) == 3

    mlflow_record = df[df["inputs"].apply(lambda x: x.get("question") == "What is MLflow?")].iloc[0]
    assert mlflow_record["inputs"]["question"] == "What is MLflow?"
    assert mlflow_record["inputs"]["context"] == "ML platforms"

    assert "expectations" in mlflow_record
    assert mlflow_record["expectations"]["correctness"] is True
    assert mlflow_record["expectations"]["completeness"] == 0.8

    assert mlflow_record["source_type"] == "TRACE"
    assert mlflow_record["source_id"] is not None

    python_record = df[df["inputs"].apply(lambda x: x.get("question") == "What is Python?")].iloc[0]
    assert python_record["expectations"]["correctness"] is True
    assert len(python_record["expectations"]) == 1

    docker_record = df[df["inputs"].apply(lambda x: x.get("question") == "What is Docker?")].iloc[0]
    assert docker_record["expectations"] is None or docker_record["expectations"] == {}

    retrieved = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    assert retrieved.tags["source"] == "trace_integration_test"
    assert retrieved.tags["version"] == "1.0"
    assert set(retrieved.experiment_ids) == {experiment}


def test_trace_deduplication_with_assessments(client, experiment):
    trace_ids = []
    for i in range(3):
        with mlflow.start_run(experiment_id=experiment):
            with mlflow.start_span(name=f"duplicate_trace_{i}") as span:
                span.set_inputs({"question": "What is AI?", "model": "gpt-4"})
                span.set_outputs({"answer": f"AI is artificial intelligence (version {i})"})
                trace_id = span.trace_id
                trace_ids.append(trace_id)

                mlflow.log_expectation(
                    trace_id=trace_id,
                    name="quality",
                    value=0.5 + i * 0.2,
                    span_id=span.span_id,
                )

    traces = [client.get_trace(tid) for tid in trace_ids]

    dataset = create_evaluation_dataset(
        name="dedup_test",
        experiment_ids=experiment,
        tags={"test": "deduplication"},
    )
    dataset.merge_records(traces)

    df = dataset.to_df()
    assert len(df) == 1

    record = df.iloc[0]
    assert record["inputs"]["question"] == "What is AI?"
    assert record["expectations"]["quality"] == 0.9
    assert record["source_id"] in trace_ids


def test_mixed_record_types_with_traces(client, experiment):
    with mlflow.start_run(experiment_id=experiment):
        with mlflow.start_span(name="mixed_test_trace") as span:
            span.set_inputs({"question": "What is ML?", "context": "machine learning"})
            span.set_outputs({"answer": "ML stands for Machine Learning"})
            trace_id = span.trace_id

            mlflow.log_expectation(
                trace_id=trace_id,
                name="accuracy",
                value=0.95,
                span_id=span.span_id,
            )

    trace = client.get_trace(trace_id)

    dataset = create_evaluation_dataset(
        name="mixed_records_test",
        experiment_ids=experiment,
        tags={"type": "mixed", "test": "true"},
    )

    manual_records = [
        {
            "inputs": {"question": "What is AI?"},
            "expectations": {"correctness": True},
            "tags": {"source": "manual"},
        },
        {
            "inputs": {"question": "What is Python?"},
            "expectations": {"correctness": True},
            "tags": {"source": "manual"},
        },
    ]
    dataset.merge_records(manual_records)

    df1 = dataset.to_df()
    assert len(df1) == 2

    dataset.merge_records([trace])

    df2 = dataset.to_df()
    assert len(df2) == 3

    ml_record = df2[df2["inputs"].apply(lambda x: x.get("question") == "What is ML?")].iloc[0]
    assert ml_record["expectations"]["accuracy"] == 0.95
    assert ml_record["source_type"] == "TRACE"

    manual_questions = {"What is AI?", "What is Python?"}
    manual_records_df = df2[df2["inputs"].apply(lambda x: x.get("question") in manual_questions)]
    assert len(manual_records_df) == 2

    for _, record in manual_records_df.iterrows():
        assert record.get("source_type") != "TRACE"


def test_trace_without_root_span_inputs(client, experiment):
    with mlflow.start_run(experiment_id=experiment):
        with mlflow.start_span(name="no_inputs_trace") as span:
            span.set_outputs({"result": "some output"})
            trace_id = span.trace_id

    trace = client.get_trace(trace_id)

    dataset = create_evaluation_dataset(
        name="no_inputs_test",
        experiment_ids=experiment,
    )

    dataset.merge_records([trace])

    df = dataset.to_df()
    assert len(df) == 1
    assert df.iloc[0]["inputs"] == {}
    assert df.iloc[0]["expectations"] is None or df.iloc[0]["expectations"] == {}


def test_error_handling_invalid_trace_types(client, experiment):
    dataset = create_evaluation_dataset(
        name="error_test",
        experiment_ids=experiment,
    )

    with mlflow.start_run(experiment_id=experiment):
        with mlflow.start_span(name="valid_trace") as span:
            span.set_inputs({"q": "test"})
            trace_id = span.trace_id

    valid_trace = client.get_trace(trace_id)

    with pytest.raises(MlflowException, match="Mixed types in trace list"):
        dataset.merge_records([valid_trace, {"inputs": {"q": "dict record"}}])

    with pytest.raises(MlflowException, match="Mixed types in trace list"):
        dataset.merge_records([valid_trace, "not a trace"])


def test_trace_integration_end_to_end(client, experiment):
    traces_to_create = [
        {
            "name": "successful_qa",
            "inputs": {"question": "What is the capital of France?", "language": "en"},
            "outputs": {"answer": "Paris", "confidence": 0.99},
            "expectations": {"correctness": True, "confidence_threshold": 0.8},
        },
        {
            "name": "incorrect_qa",
            "inputs": {"question": "What is 2+2?", "language": "en"},
            "outputs": {"answer": "5", "confidence": 0.5},
            "expectations": {"correctness": False},
        },
        {
            "name": "multilingual_qa",
            "inputs": {"question": "¿Cómo estás?", "language": "es"},
            "outputs": {"answer": "I'm doing well, thank you!", "confidence": 0.9},
            "expectations": {"language_match": False, "politeness": True},
        },
    ]

    created_trace_ids = []
    for trace_config in traces_to_create:
        with mlflow.start_run(experiment_id=experiment):
            with mlflow.start_span(name=trace_config["name"]) as span:
                span.set_inputs(trace_config["inputs"])
                span.set_outputs(trace_config["outputs"])
                span.set_attributes(
                    {
                        "model": "test-llm-v1",
                        "temperature": 0.7,
                        "max_tokens": 100,
                    }
                )
                trace_id = span.trace_id
                created_trace_ids.append(trace_id)

                for exp_name, exp_value in trace_config["expectations"].items():
                    mlflow.log_expectation(
                        trace_id=trace_id,
                        name=exp_name,
                        value=exp_value,
                        span_id=span.span_id,
                        metadata={"trace_name": trace_config["name"]},
                    )

    dataset = create_evaluation_dataset(
        name="comprehensive_trace_test",
        experiment_ids=[experiment],
        tags={
            "test_type": "end_to_end",
            "model": "test-llm-v1",
            "language": "multilingual",
        },
    )

    traces = [client.get_trace(tid) for tid in created_trace_ids]
    dataset.merge_records(traces)

    df = dataset.to_df()
    assert len(df) == 3

    french_record = df[df["inputs"].apply(lambda x: "France" in str(x.get("question", "")))].iloc[0]
    assert french_record["expectations"]["correctness"] is True
    assert french_record["expectations"]["confidence_threshold"] == 0.8

    math_record = df[df["inputs"].apply(lambda x: "2+2" in str(x.get("question", "")))].iloc[0]
    assert math_record["expectations"]["correctness"] is False

    spanish_record = df[df["inputs"].apply(lambda x: x.get("language") == "es")].iloc[0]
    assert spanish_record["expectations"]["language_match"] is False
    assert spanish_record["expectations"]["politeness"] is True

    retrieved_dataset = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    retrieved_df = retrieved_dataset.to_df()
    assert len(retrieved_df) == 3
    assert retrieved_dataset.tags["model"] == "test-llm-v1"

    additional_records = [
        {
            "inputs": {"question": "What is Python?", "language": "en"},
            "expectations": {"technical_accuracy": True},
            "tags": {"source": "manual_addition"},
        }
    ]
    retrieved_dataset.merge_records(additional_records)

    final_df = retrieved_dataset.to_df()
    assert len(final_df) == 4

    trace_records = final_df[final_df["source_type"] == "TRACE"]
    assert len(trace_records) == 3

    manual_records = final_df[final_df["source_type"] != "TRACE"]
    assert len(manual_records) == 1


def test_evaluation_dataset_tags_crud_workflow(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="test_tags_crud",
        experiment_ids=experiments[0],
    )
    initial_tags = dataset.tags.copy()

    set_evaluation_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={
            "team": "ml-platform",
            "project": "evaluation",
            "priority": "high",
        },
    )

    dataset = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    expected_tags = initial_tags.copy()
    expected_tags.update(
        {
            "team": "ml-platform",
            "project": "evaluation",
            "priority": "high",
        }
    )
    assert dataset.tags == expected_tags

    set_evaluation_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={
            "priority": "medium",
            "status": "active",
        },
    )

    dataset = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    expected_tags = initial_tags.copy()
    expected_tags.update(
        {
            "team": "ml-platform",
            "project": "evaluation",
            "priority": "medium",
            "status": "active",
        }
    )
    assert dataset.tags == expected_tags

    delete_evaluation_dataset_tag(
        dataset_id=dataset.dataset_id,
        key="priority",
    )

    dataset = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    expected_tags = initial_tags.copy()
    expected_tags.update(
        {
            "team": "ml-platform",
            "project": "evaluation",
            "status": "active",
        }
    )
    assert dataset.tags == expected_tags

    delete_evaluation_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        get_evaluation_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        set_evaluation_dataset_tags(
            dataset_id=dataset.dataset_id,
            tags={"should": "fail"},
        )

    delete_evaluation_dataset_tag(dataset_id=dataset.dataset_id, key="status")


def test_set_evaluation_dataset_tags_databricks(mock_databricks_environment):
    with pytest.raises(NotImplementedError, match="tag operations are not available"):
        set_evaluation_dataset_tags(dataset_id="test", tags={"key": "value"})


def test_delete_evaluation_dataset_tag_databricks(mock_databricks_environment):
    with pytest.raises(NotImplementedError, match="tag operations are not available"):
        delete_evaluation_dataset_tag(dataset_id="test", key="key")


def test_evaluation_dataset_schema_evolution_and_log_input(tracking_uri, experiments):
    dataset = create_evaluation_dataset(
        name="schema_evolution_test",
        experiment_ids=[experiments[0]],
        tags={"test": "schema_evolution", "mlflow.user": "test_user"},
    )

    stage1_records = [
        {
            "inputs": {"prompt": "What is MLflow?"},
            "expectations": {"response": "MLflow is a platform"},
        }
    ]
    dataset.merge_records(stage1_records)

    ds1 = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    schema1 = json.loads(ds1.schema)
    assert schema1 is not None
    assert "prompt" in schema1["inputs"]
    assert schema1["inputs"]["prompt"] == "string"
    assert len(schema1["inputs"]) == 1
    assert len(schema1["expectations"]) == 1

    stage2_records = [
        {
            "inputs": {
                "prompt": "Explain Python",
                "temperature": 0.7,
                "max_length": 500,
                "top_p": 0.95,
            },
            "expectations": {
                "response": "Python is a programming language",
                "quality_score": 0.85,
                "token_count": 127,
            },
        }
    ]
    dataset.merge_records(stage2_records)

    ds2 = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    schema2 = json.loads(ds2.schema)
    assert "temperature" in schema2["inputs"]
    assert schema2["inputs"]["temperature"] == "float"
    assert "max_length" in schema2["inputs"]
    assert schema2["inputs"]["max_length"] == "integer"
    assert len(schema2["inputs"]) == 4
    assert len(schema2["expectations"]) == 3

    stage3_records = [
        {
            "inputs": {
                "prompt": "Complex query",
                "streaming": True,
                "stop_sequences": ["\n\n", "END"],
                "config": {"model": "gpt-4", "version": "1.0"},
            },
            "expectations": {
                "response": "Complex response",
                "is_valid": True,
                "citations": ["source1", "source2"],
                "metadata": {"confidence": 0.9},
            },
        }
    ]
    dataset.merge_records(stage3_records)

    ds3 = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    schema3 = json.loads(ds3.schema)

    assert schema3["inputs"]["streaming"] == "boolean"
    assert schema3["inputs"]["stop_sequences"] == "array"
    assert schema3["inputs"]["config"] == "object"
    assert schema3["expectations"]["is_valid"] == "boolean"
    assert schema3["expectations"]["citations"] == "array"
    assert schema3["expectations"]["metadata"] == "object"

    assert "prompt" in schema3["inputs"]
    assert "temperature" in schema3["inputs"]
    assert "quality_score" in schema3["expectations"]

    with mlflow.start_run(experiment_id=experiments[0]) as run:
        mlflow.log_input(dataset, context="evaluation")

        mlflow.log_metrics({"accuracy": 0.92, "f1_score": 0.89})

    run_data = mlflow.get_run(run.info.run_id)
    assert run_data.inputs is not None
    assert run_data.inputs.dataset_inputs is not None
    assert len(run_data.inputs.dataset_inputs) > 0

    dataset_input = run_data.inputs.dataset_inputs[0]
    assert dataset_input.dataset.name == "schema_evolution_test"
    assert dataset_input.dataset.source_type == "mlflow_evaluation_dataset"

    tag_dict = {tag.key: tag.value for tag in dataset_input.tags}
    assert "mlflow.data.context" in tag_dict
    assert tag_dict["mlflow.data.context"] == "evaluation"

    final_dataset = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    final_schema = json.loads(final_dataset.schema)

    assert "inputs" in final_schema
    assert "expectations" in final_schema
    assert "version" in final_schema
    assert final_schema["version"] == "1.0"

    profile = json.loads(final_dataset.profile)
    assert profile is not None
    assert profile["num_records"] == 3

    consistency_records = [
        {
            "inputs": {"prompt": "Another test", "temperature": 0.5, "max_length": 200},
            "expectations": {"response": "Another response", "quality_score": 0.75},
        }
    ]
    dataset.merge_records(consistency_records)

    consistent_dataset = get_evaluation_dataset(dataset_id=dataset.dataset_id)
    consistent_schema = json.loads(consistent_dataset.schema)

    assert set(consistent_schema["inputs"].keys()) == set(final_schema["inputs"].keys())
    assert set(consistent_schema["expectations"].keys()) == set(final_schema["expectations"].keys())

    consistent_profile = json.loads(consistent_dataset.profile)
    assert consistent_profile["num_records"] == 4
