import json
import os
import sys
import warnings
from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.data import Dataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin
from mlflow.entities.dataset_record_source import DatasetRecordSourceType
from mlflow.entities.evaluation_dataset import EvaluationDataset as EntityEvaluationDataset
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import (
    EvaluationDataset,
    create_dataset,
    delete_dataset,
    delete_dataset_tag,
    get_dataset,
    search_datasets,
    set_dataset_tags,
)
from mlflow.genai.datasets.evaluation_dataset import (
    EvaluationDataset as WrapperEvaluationDataset,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_USER


@pytest.fixture
def mock_client():
    with mock.patch("mlflow.tracking.client.MlflowClient") as mock_client_class:
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


def test_create_dataset(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
        tags={"environment": "production", "version": "1.0"},
    )
    mock_client.create_dataset.return_value = expected_dataset

    result = create_dataset(
        name="test_dataset",
        experiment_id=["exp1", "exp2"],
        tags={"environment": "production", "version": "1.0"},
    )

    assert result == expected_dataset
    mock_client.create_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_id=["exp1", "exp2"],
        tags={"environment": "production", "version": "1.0"},
    )


def test_create_dataset_single_experiment_id(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
    )
    mock_client.create_dataset.return_value = expected_dataset

    result = create_dataset(
        name="test_dataset",
        experiment_id="exp1",
    )

    assert result == expected_dataset
    mock_client.create_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_id=["exp1"],
        tags=None,
    )


def test_create_dataset_with_empty_tags(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
        tags={},
    )
    mock_client.create_dataset.return_value = expected_dataset

    result = create_dataset(
        name="test_dataset",
        experiment_id=["exp1"],
        tags={},
    )

    assert result == expected_dataset
    mock_client.create_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_id=["exp1"],
        tags={},
    )


def test_create_dataset_databricks(mock_databricks_environment):
    mock_dataset = mock.Mock()
    with mock.patch.dict(
        "sys.modules",
        {
            "databricks.agents.datasets": mock.Mock(
                create_dataset=mock.Mock(return_value=mock_dataset)
            )
        },
    ):
        result = create_dataset(
            name="catalog.schema.table",
            experiment_id=["exp1", "exp2"],
        )

        sys.modules["databricks.agents.datasets"].create_dataset.assert_called_once_with(
            "catalog.schema.table", ["exp1", "exp2"]
        )
        assert isinstance(result, EvaluationDataset)


def test_get_dataset(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        digest="abc123",
        created_time=123456789,
        last_update_time=123456789,
    )
    mock_client.get_dataset.return_value = expected_dataset

    result = get_dataset(dataset_id="test_id")

    assert result == expected_dataset
    mock_client.get_dataset.assert_called_once_with("test_id")


def test_get_dataset_missing_id():
    with pytest.raises(ValueError, match="Parameter 'dataset_id' is required"):
        get_dataset()


def test_get_dataset_databricks(mock_databricks_environment):
    mock_dataset = mock.Mock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.agents.datasets": mock.Mock(get_dataset=mock.Mock(return_value=mock_dataset))},
    ):
        result = get_dataset(name="catalog.schema.table")

        sys.modules["databricks.agents.datasets"].get_dataset.assert_called_once_with(
            "catalog.schema.table"
        )
        assert isinstance(result, EvaluationDataset)


def test_get_dataset_databricks_missing_name(mock_databricks_environment):
    with pytest.raises(ValueError, match="Parameter 'name' is required"):
        get_dataset(dataset_id="test_id")


def test_delete_dataset(mock_client):
    delete_dataset(dataset_id="test_id")

    mock_client.delete_dataset.assert_called_once_with("test_id")


def test_delete_dataset_missing_id():
    with pytest.raises(ValueError, match="Parameter 'dataset_id' is required"):
        delete_dataset()


def test_delete_dataset_databricks(mock_databricks_environment):
    with mock.patch.dict(
        "sys.modules", {"databricks.agents.datasets": mock.Mock(delete_dataset=mock.Mock())}
    ):
        delete_dataset(name="catalog.schema.table")

        sys.modules["databricks.agents.datasets"].delete_dataset.assert_called_once_with(
            "catalog.schema.table"
        )


def test_search_datasets_with_mock(mock_client):
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
    mock_client.search_datasets.return_value = PagedList(datasets, None)

    result = search_datasets(
        experiment_ids=["exp1", "exp2"],
        filter_string="name LIKE 'test%'",
        max_results=100,
        order_by=["created_time DESC"],
    )

    assert len(result) == 2
    assert isinstance(result, list)

    mock_client.search_datasets.assert_called_once_with(
        experiment_ids=["exp1", "exp2"],
        filter_string="name LIKE 'test%'",
        max_results=50,
        order_by=["created_time DESC"],
        page_token=None,
    )


def test_search_datasets_single_experiment_id(mock_client):
    datasets = [
        EntityEvaluationDataset(
            dataset_id="id1",
            name="dataset1",
            digest="digest1",
            created_time=123456789,
            last_update_time=123456789,
        )
    ]
    mock_client.search_datasets.return_value = PagedList(datasets, None)

    # When no max_results is specified, it defaults to None which means get all
    # Mock time to have a consistent filter_string
    with mock.patch("time.time", return_value=1234567890):
        search_datasets(experiment_ids="exp1")

    # The pagination wrapper will use SEARCH_EVALUATION_DATASETS_MAX_RESULTS as the page size
    # Now the function adds default filter (last 7 days) and order_by when not specified
    seven_days_ago = int((1234567890 - 7 * 24 * 60 * 60) * 1000)
    mock_client.search_datasets.assert_called_once_with(
        experiment_ids=["exp1"],
        filter_string=f"created_time >= {seven_days_ago}",
        max_results=SEARCH_EVALUATION_DATASETS_MAX_RESULTS,  # Page size
        order_by=["created_time DESC"],
        page_token=None,
    )


def test_search_datasets_pagination_handling(mock_client):
    page1_datasets = [
        EntityEvaluationDataset(
            dataset_id=f"id{i}",
            name=f"dataset{i}",
            digest=f"digest{i}",
            created_time=123456789,
            last_update_time=123456789,
        )
        for i in range(3)
    ]

    page2_datasets = [
        EntityEvaluationDataset(
            dataset_id=f"id{i}",
            name=f"dataset{i}",
            digest=f"digest{i}",
            created_time=123456789,
            last_update_time=123456789,
        )
        for i in range(3, 5)
    ]

    mock_client.search_datasets.side_effect = [
        PagedList(page1_datasets, "token1"),
        PagedList(page2_datasets, None),
    ]

    result = search_datasets(experiment_ids=["exp1"], max_results=10)

    assert len(result) == 5
    assert isinstance(result, list)

    assert mock_client.search_datasets.call_count == 2

    first_call = mock_client.search_datasets.call_args_list[0]
    assert first_call[1]["page_token"] is None

    second_call = mock_client.search_datasets.call_args_list[1]
    assert second_call[1]["page_token"] == "token1"


def test_search_datasets_single_page(mock_client):
    datasets = [
        EntityEvaluationDataset(
            dataset_id="id1",
            name="dataset1",
            digest="digest1",
            created_time=123456789,
            last_update_time=123456789,
        )
    ]

    mock_client.search_datasets.return_value = PagedList(datasets, None)

    result = search_datasets(max_results=10)

    assert len(result) == 1
    assert isinstance(result, list)

    assert mock_client.search_datasets.call_count == 1


def test_search_datasets_databricks(mock_databricks_environment):
    with pytest.raises(NotImplementedError, match="Dataset search is not available in Databricks"):
        search_datasets()


def test_databricks_import_error():
    with mock.patch("mlflow.genai.datasets.is_databricks_default_tracking_uri", return_value=True):
        with mock.patch.dict("sys.modules", {"databricks.agents.datasets": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="databricks-agents"):
                    create_dataset(name="test", experiment_id="exp1")


def test_create_dataset_with_user_tag(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_user_attribution",
        experiment_id=experiments[0],
        tags={"environment": "test", MLFLOW_USER: "john_doe"},
    )

    assert dataset.name == "test_user_attribution"
    assert dataset.tags[MLFLOW_USER] == "john_doe"
    assert dataset.created_by == "john_doe"

    dataset2 = create_dataset(
        name="test_no_user",
        experiment_id=experiments[0],
        tags={"environment": "test"},
    )

    assert dataset2.name == "test_no_user"
    assert isinstance(dataset2.tags[MLFLOW_USER], str)
    assert dataset2.created_by == dataset2.tags[MLFLOW_USER]


def test_create_and_get_dataset(tracking_uri, experiments):
    dataset = create_dataset(
        name="qa_evaluation_v1",
        experiment_id=[experiments[0], experiments[1]],
        tags={"source": "manual_curation", "environment": "test"},
    )

    assert dataset.name == "qa_evaluation_v1"
    assert dataset.tags["source"] == "manual_curation"
    assert dataset.tags["environment"] == "test"
    assert len(dataset.experiment_ids) == 2
    assert dataset.dataset_id is not None

    retrieved = get_dataset(dataset_id=dataset.dataset_id)

    assert retrieved.dataset_id == dataset.dataset_id
    assert retrieved.name == dataset.name
    assert retrieved.tags == dataset.tags
    assert set(retrieved.experiment_ids) == {experiments[0], experiments[1]}


def test_create_dataset_minimal_params(tracking_uri):
    dataset = create_dataset(name="minimal_dataset")

    assert dataset.name == "minimal_dataset"
    assert "mlflow.user" not in dataset.tags or isinstance(dataset.tags.get("mlflow.user"), str)
    assert dataset.experiment_ids == ["0"]


def test_active_record_pattern_merge_records(tracking_uri, experiments):
    dataset = create_dataset(
        name="active_record_test",
        experiment_id=experiments[0],
    )

    records_batch1 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": {
                "answer": "MLflow is an open source platform for managing the ML lifecycle",
                "key1": "value1",
            },
            "expectations": {
                "answer": "MLflow is an open source platform",
                "key2": "value2",
            },
            "tags": {"difficulty": "easy"},
        },
        {
            "inputs": {"question": "What is Python?"},
            "outputs": {"answer": "Python is a versatile programming language"},
            "expectations": {"answer": "Python is a programming language"},
            "tags": {"difficulty": "easy"},
        },
    ]

    records_batch2 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "outputs": {"answer": "MLflow is a popular ML lifecycle platform"},
            "expectations": {"answer": "MLflow is an ML lifecycle platform"},
            "tags": {"category": "ml"},
        },
        {
            "inputs": {"question": "What is Docker?"},
            "outputs": {"answer": "Docker is a popular containerization platform"},
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
    assert mlflow_record["expectations"] == {
        "answer": "MLflow is an open source platform",
        "key2": "value2",
    }
    assert mlflow_record["outputs"] == {
        "answer": "MLflow is an open source platform for managing the ML lifecycle",
        "key1": "value1",
    }
    assert mlflow_record["tags"]["difficulty"] == "easy"
    assert "category" not in mlflow_record["tags"]

    dataset.merge_records(records_batch2)

    df2 = dataset.to_df()
    assert len(df2) == 3

    mlflow_record_updated = df2[
        df2["inputs"].apply(lambda x: x.get("question") == "What is MLflow?")
    ].iloc[0]

    assert mlflow_record_updated["expectations"] == {
        "answer": "MLflow is an ML lifecycle platform",
        "key2": "value2",
    }
    assert mlflow_record_updated["outputs"] == {
        "answer": "MLflow is a popular ML lifecycle platform"
    }
    assert mlflow_record_updated["tags"]["difficulty"] == "easy"
    assert mlflow_record_updated["tags"]["category"] == "ml"

    # Verify that the new Docker record also has outputs
    docker_record = df2[df2["inputs"].apply(lambda x: x.get("question") == "What is Docker?")].iloc[
        0
    ]
    assert docker_record["outputs"]["answer"] == "Docker is a popular containerization platform"
    assert docker_record["expectations"]["answer"] == "Docker is a containerization platform"
    assert docker_record["tags"]["difficulty"] == "medium"


def test_dataset_with_dataframe_records(tracking_uri, experiments):
    dataset = create_dataset(
        name="dataframe_test",
        experiment_id=experiments[0],
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

    # Check that all expected records are present (order-agnostic)
    texts = {record["inputs"]["text"] for _, record in result_df.iterrows()}
    expected_texts = {"The movie was amazing!", "Terrible experience"}
    assert texts == expected_texts

    sentiments = {record["expectations"]["sentiment"] for _, record in result_df.iterrows()}
    expected_sentiments = {"positive", "negative"}
    assert sentiments == expected_sentiments


def test_search_datasets(tracking_uri, experiments):
    datasets = []
    for i in range(5):
        dataset = create_dataset(
            name=f"search_test_{i}",
            experiment_id=[experiments[i % len(experiments)]],
            tags={"type": "human" if i % 2 == 0 else "trace", "index": str(i)},
        )
        datasets.append(dataset)

    all_results = search_datasets()
    assert len(all_results) == 5

    exp0_results = search_datasets(experiment_ids=experiments[0])
    assert len(exp0_results) == 2

    human_results = search_datasets(filter_string="name LIKE 'search_test_%'")
    assert len(human_results) == 5

    limited_results = search_datasets(max_results=2)
    assert len(limited_results) == 2

    more_results = search_datasets(max_results=4)
    assert len(more_results) == 4


def test_delete_dataset(tracking_uri, experiments):
    dataset = create_dataset(
        name="to_be_deleted",
        experiment_id=[experiments[0], experiments[1]],
        tags={"env": "test", "version": "1.0"},
    )
    dataset_id = dataset.dataset_id

    dataset.merge_records([{"inputs": {"q": "test"}, "expectations": {"a": "answer"}}])

    retrieved = get_dataset(dataset_id=dataset_id)
    assert retrieved is not None
    assert len(retrieved.to_df()) == 1

    delete_dataset(dataset_id=dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        get_dataset(dataset_id=dataset_id)

    search_results = search_datasets(experiment_ids=[experiments[0], experiments[1]])
    found_ids = [d.dataset_id for d in search_results]
    assert dataset_id not in found_ids


def test_dataset_lifecycle_workflow(tracking_uri, experiments):
    dataset = create_dataset(
        name="qa_eval_prod_v1",
        experiment_id=[experiments[0], experiments[1]],
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
    retrieved = get_dataset(dataset_id=dataset_id)
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

    found = search_datasets(
        experiment_ids=experiments[0],
        filter_string="name LIKE 'qa_eval%'",
    )
    assert len(found) == 1
    assert found[0].dataset_id == dataset_id

    final_dataset = get_dataset(dataset_id=dataset_id)
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
        create_dataset(name="test")
    assert exc.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        get_dataset(dataset_id="test_id")
    assert exc.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        search_datasets()
    assert exc.value.error_code == "FEATURE_DISABLED"

    with pytest.raises(MlflowException, match="not supported with FileStore") as exc:
        delete_dataset(dataset_id="test_id")
    assert exc.value.error_code == "FEATURE_DISABLED"


def test_single_experiment_id_handling(tracking_uri, experiments):
    dataset = create_dataset(
        name="single_exp_test",
        experiment_id=experiments[0],
    )

    assert isinstance(dataset.experiment_ids, list)
    assert dataset.experiment_ids == [experiments[0]]

    results = search_datasets(experiment_ids=experiments[0])
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

                mlflow.log_expectation(
                    trace_id=trace_id,
                    name="expected_answer",
                    value=f"Detailed answer for {inputs['question']}",
                )
                mlflow.log_expectation(
                    trace_id=trace_id,
                    name="quality_score",
                    value=0.85 + i * 0.05,
                )

    traces = mlflow.search_traces(
        experiment_ids=[experiments[0], experiments[1]],
        max_results=10,
        return_type="list",
    )
    assert len(traces) == 3

    dataset = create_dataset(
        name="trace_eval_dataset",
        experiment_id=[experiments[0], experiments[1]],
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

    retrieved = get_dataset(dataset_id=dataset.dataset_id)
    retrieved_df = retrieved.to_df()
    assert len(retrieved_df) == 3

    delete_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        get_dataset(dataset_id=dataset.dataset_id)

    search_results = search_datasets(
        experiment_ids=[experiments[0], experiments[1]], max_results=100
    )
    found_dataset_ids = [d.dataset_id for d in search_results]
    assert dataset.dataset_id not in found_dataset_ids

    all_datasets = search_datasets(max_results=100)
    all_dataset_ids = [d.dataset_id for d in all_datasets]
    assert dataset.dataset_id not in all_dataset_ids


def test_search_traces_dataframe_to_dataset_integration(tracking_uri, experiments):
    for i in range(3):
        with mlflow.start_run(experiment_id=experiments[0]):
            with mlflow.start_span(name=f"test_span_{i}") as span:
                span.set_inputs({"question": f"Question {i}?", "temperature": 0.7})
                span.set_outputs({"answer": f"Answer {i}"})

                mlflow.log_expectation(
                    trace_id=span.trace_id,
                    name="expected_answer",
                    value=f"Expected answer {i}",
                )
                mlflow.log_expectation(
                    trace_id=span.trace_id,
                    name="min_score",
                    value=0.8,
                )

    traces_df = mlflow.search_traces(
        experiment_ids=[experiments[0]],
    )

    assert "trace" in traces_df.columns
    assert "assessments" in traces_df.columns
    assert len(traces_df) == 3

    dataset = create_dataset(
        name="traces_dataframe_dataset",
        experiment_id=experiments[0],
        tags={"source": "search_traces", "format": "dataframe"},
    )

    dataset.merge_records(traces_df)

    result_df = dataset.to_df()
    assert len(result_df) == 3

    for idx, row in result_df.iterrows():
        assert "inputs" in row
        assert "expectations" in row
        assert "source_type" in row
        assert row["source_type"] == "TRACE"

        assert "question" in row["inputs"]
        question_text = row["inputs"]["question"]
        assert question_text.startswith("Question ")
        assert question_text.endswith("?")
        question_num = int(question_text.replace("Question ", "").replace("?", ""))
        assert 0 <= question_num <= 2

        assert "expected_answer" in row["expectations"]
        assert f"Expected answer {question_num}" == row["expectations"]["expected_answer"]
        assert "min_score" in row["expectations"]
        assert row["expectations"]["min_score"] == 0.8


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

    dataset = create_dataset(
        name="trace_assessment_dataset",
        experiment_id=[experiment],
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

    retrieved = get_dataset(dataset_id=dataset.dataset_id)
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

    dataset = create_dataset(
        name="dedup_test",
        experiment_id=experiment,
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

    dataset = create_dataset(
        name="mixed_records_test",
        experiment_id=experiment,
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

    dataset = create_dataset(
        name="no_inputs_test",
        experiment_id=experiment,
    )

    dataset.merge_records([trace])

    df = dataset.to_df()
    assert len(df) == 1
    assert df.iloc[0]["inputs"] == {}
    assert df.iloc[0]["expectations"] is None or df.iloc[0]["expectations"] == {}


def test_error_handling_invalid_trace_types(client, experiment):
    dataset = create_dataset(
        name="error_test",
        experiment_id=experiment,
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

    dataset = create_dataset(
        name="comprehensive_trace_test",
        experiment_id=[experiment],
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

    retrieved_dataset = get_dataset(dataset_id=dataset.dataset_id)
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


def test_dataset_pagination_transparency_large_records(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_pagination_transparency",
        experiment_id=experiments[0],
        tags={"test": "large_dataset"},
    )

    large_records = []
    for i in range(150):
        large_records.append(
            {
                "inputs": {"question": f"Question {i}", "index": i},
                "expectations": {"answer": f"Answer {i}", "score": i * 0.01},
            }
        )

    dataset.merge_records(large_records)

    all_records = dataset.records
    assert len(all_records) == 150

    record_indices = {record.inputs["index"] for record in all_records}
    expected_indices = set(range(150))
    assert record_indices == expected_indices

    record_scores = {record.expectations["score"] for record in all_records}
    expected_scores = {i * 0.01 for i in range(150)}
    assert record_scores == expected_scores

    df = dataset.to_df()
    assert len(df) == 150

    df_indices = {row["index"] for row in df["inputs"]}
    assert df_indices == expected_indices

    assert not hasattr(dataset, "page_token")
    assert not hasattr(dataset, "next_page_token")
    assert not hasattr(dataset, "max_results")

    second_access = dataset.records
    assert second_access is all_records

    dataset._records = None
    refreshed_records = dataset.records
    assert len(refreshed_records) == 150


def test_dataset_internal_pagination_with_mock(tracking_uri, experiments):
    from mlflow.tracking._tracking_service.utils import _get_store

    dataset = create_dataset(
        name="test_internal_pagination",
        experiment_id=experiments[0],
        tags={"test": "pagination_mock"},
    )

    records = []
    for i in range(75):
        records.append(
            {"inputs": {"question": f"Q{i}", "id": i}, "expectations": {"answer": f"A{i}"}}
        )

    dataset.merge_records(records)

    dataset._records = None

    store = _get_store()
    with mock.patch.object(
        store, "_load_dataset_records", wraps=store._load_dataset_records
    ) as mock_load:
        accessed_records = dataset.records

        mock_load.assert_called_once_with(dataset.dataset_id, max_results=None)
        assert len(accessed_records) == 75

    dataset._records = None

    with mock.patch.object(
        store, "_load_dataset_records", wraps=store._load_dataset_records
    ) as mock_load:
        df = dataset.to_df()

        mock_load.assert_called_once_with(dataset.dataset_id, max_results=None)
        assert len(df) == 75


def test_dataset_experiment_associations(tracking_uri, experiments):
    from mlflow.genai.datasets import (
        add_dataset_to_experiments,
        remove_dataset_from_experiments,
    )

    dataset = create_dataset(
        name="test_associations",
        experiment_id=experiments[0],
        tags={"test": "associations"},
    )

    initial_exp_ids = dataset.experiment_ids
    assert experiments[0] in initial_exp_ids

    updated = add_dataset_to_experiments(
        dataset_id=dataset.dataset_id, experiment_ids=[experiments[1], experiments[2]]
    )
    assert experiments[0] in updated.experiment_ids
    assert experiments[1] in updated.experiment_ids
    assert experiments[2] in updated.experiment_ids
    assert len(updated.experiment_ids) == 3

    result = add_dataset_to_experiments(
        dataset_id=dataset.dataset_id, experiment_ids=[experiments[1], experiments[2]]
    )
    assert len(result.experiment_ids) == 3
    assert all(exp in result.experiment_ids for exp in experiments)

    removed = remove_dataset_from_experiments(
        dataset_id=dataset.dataset_id, experiment_ids=[experiments[1], experiments[2]]
    )
    assert experiments[1] not in removed.experiment_ids
    assert experiments[2] not in removed.experiment_ids
    assert experiments[0] in removed.experiment_ids
    assert len(removed.experiment_ids) == 1

    with mock.patch("mlflow.store.tracking.sqlalchemy_store._logger.warning") as mock_warning:
        idempotent = remove_dataset_from_experiments(
            dataset_id=dataset.dataset_id, experiment_ids=[experiments[1], experiments[2]]
        )
        assert mock_warning.call_count == 2
        assert "was not associated" in mock_warning.call_args_list[0][0][0]

    assert len(idempotent.experiment_ids) == 1


def test_dataset_associations_filestore_blocking(tmp_path):
    from mlflow.genai.datasets import (
        add_dataset_to_experiments,
        remove_dataset_from_experiments,
    )

    mlflow.set_tracking_uri(tmp_path.as_uri())

    with pytest.raises(NotImplementedError, match="not supported with FileStore"):
        add_dataset_to_experiments(dataset_id="d-test123", experiment_ids=["1", "2"])

    with pytest.raises(NotImplementedError, match="not supported with FileStore"):
        remove_dataset_from_experiments(dataset_id="d-test123", experiment_ids=["1"])


def test_evaluation_dataset_tags_crud_workflow(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_tags_crud",
        experiment_id=experiments[0],
    )
    initial_tags = dataset.tags.copy()

    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={
            "team": "ml-platform",
            "project": "evaluation",
            "priority": "high",
        },
    )

    dataset = get_dataset(dataset_id=dataset.dataset_id)
    expected_tags = initial_tags.copy()
    expected_tags.update(
        {
            "team": "ml-platform",
            "project": "evaluation",
            "priority": "high",
        }
    )
    assert dataset.tags == expected_tags

    set_dataset_tags(
        dataset_id=dataset.dataset_id,
        tags={
            "priority": "medium",
            "status": "active",
        },
    )

    dataset = get_dataset(dataset_id=dataset.dataset_id)
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

    delete_dataset_tag(
        dataset_id=dataset.dataset_id,
        key="priority",
    )

    dataset = get_dataset(dataset_id=dataset.dataset_id)
    expected_tags = initial_tags.copy()
    expected_tags.update(
        {
            "team": "ml-platform",
            "project": "evaluation",
            "status": "active",
        }
    )
    assert dataset.tags == expected_tags

    delete_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        get_dataset(dataset_id=dataset.dataset_id)

    with pytest.raises(MlflowException, match="Could not find|not found"):
        set_dataset_tags(
            dataset_id=dataset.dataset_id,
            tags={"should": "fail"},
        )

    delete_dataset_tag(dataset_id=dataset.dataset_id, key="status")


def test_set_dataset_tags_databricks(mock_databricks_environment):
    with pytest.raises(NotImplementedError, match="tag operations are not available"):
        set_dataset_tags(dataset_id="test", tags={"key": "value"})


def test_delete_dataset_tag_databricks(mock_databricks_environment):
    with pytest.raises(NotImplementedError, match="tag operations are not available"):
        delete_dataset_tag(dataset_id="test", key="key")


def test_dataset_schema_evolution_and_log_input(tracking_uri, experiments):
    dataset = create_dataset(
        name="schema_evolution_test",
        experiment_id=[experiments[0]],
        tags={"test": "schema_evolution", "mlflow.user": "test_user"},
    )

    stage1_records = [
        {
            "inputs": {"prompt": "What is MLflow?"},
            "expectations": {"response": "MLflow is a platform"},
        }
    ]
    dataset.merge_records(stage1_records)

    ds1 = get_dataset(dataset_id=dataset.dataset_id)
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

    ds2 = get_dataset(dataset_id=dataset.dataset_id)
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

    ds3 = get_dataset(dataset_id=dataset.dataset_id)
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

    final_dataset = get_dataset(dataset_id=dataset.dataset_id)
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

    consistent_dataset = get_dataset(dataset_id=dataset.dataset_id)
    consistent_schema = json.loads(consistent_dataset.schema)

    assert set(consistent_schema["inputs"].keys()) == set(final_schema["inputs"].keys())
    assert set(consistent_schema["expectations"].keys()) == set(final_schema["expectations"].keys())

    consistent_profile = json.loads(consistent_dataset.profile)
    assert consistent_profile["num_records"] == 4
    delete_dataset_tag(dataset_id="test", key="key")


def test_deprecated_parameter_substitution(experiment):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        dataset = create_dataset(
            uc_table_name="test_dataset_deprecated",
            experiment_id=experiment,
            tags={"test": "deprecated_parameter"},
        )

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "uc_table_name" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()
        assert "name" in str(w[0].message)

        assert dataset.name == "test_dataset_deprecated"
        assert dataset.tags["test"] == "deprecated_parameter"

    with pytest.raises(ValueError, match="Cannot specify both.*uc_table_name.*and.*name"):
        create_dataset(
            uc_table_name="old_name",
            name="new_name",
            experiment_id=experiment,
        )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        with pytest.raises(ValueError, match="name.*only supported in Databricks"):
            get_dataset(uc_table_name="test_dataset_deprecated")

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "uc_table_name" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        with pytest.raises(ValueError, match="name.*only supported in Databricks"):
            delete_dataset(uc_table_name="test_dataset_deprecated")

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "uc_table_name" in str(w[0].message)

    delete_dataset(dataset_id=dataset.dataset_id)


def test_create_dataset_uses_active_experiment_when_not_specified(tracking_uri):
    exp_id = mlflow.create_experiment("test_active_experiment")
    mlflow.set_experiment(experiment_id=exp_id)

    dataset = create_dataset(name="test_with_active_exp")

    assert dataset.experiment_ids == [exp_id]

    from mlflow.tracking import fluent

    fluent._active_experiment_id = None


def test_create_dataset_with_no_active_experiment(tracking_uri):
    from mlflow.tracking import fluent

    fluent._active_experiment_id = None

    dataset = create_dataset(name="test_no_active_exp")

    assert dataset.experiment_ids == ["0"]


def test_create_dataset_explicit_overrides_active_experiment(tracking_uri):
    active_exp = mlflow.create_experiment("active_exp")
    explicit_exp = mlflow.create_experiment("explicit_exp")

    mlflow.set_experiment(experiment_id=active_exp)

    dataset = create_dataset(name="test_explicit_override", experiment_id=explicit_exp)

    assert dataset.experiment_ids == [explicit_exp]

    from mlflow.tracking import fluent

    fluent._active_experiment_id = None


def test_create_dataset_none_uses_active_experiment(tracking_uri):
    exp_id = mlflow.create_experiment("test_none_experiment")
    mlflow.set_experiment(experiment_id=exp_id)

    dataset = create_dataset(name="test_none_exp", experiment_id=None)

    assert dataset.experiment_ids == [exp_id]

    from mlflow.tracking import fluent

    fluent._active_experiment_id = None


def test_source_type_inference():
    exp = mlflow.create_experiment("test_source_inference")
    dataset = create_dataset(
        name="test_source_inference", experiment_id=exp, tags={"test": "source_inference"}
    )

    human_records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform", "quality": 0.9},
        },
        {
            "inputs": {"question": "How to track experiments?"},
            "expectations": {"answer": "Use mlflow.start_run()", "quality": 0.85},
        },
    ]
    dataset.merge_records(human_records)

    df = dataset.to_df()
    human_sources = df[df["source_type"] == DatasetRecordSourceType.HUMAN.value]
    assert len(human_sources) == 2

    code_records = [{"inputs": {"question": f"Generated question {i}"}} for i in range(3)]
    dataset.merge_records(code_records)

    df = dataset.to_df()
    code_sources = df[df["source_type"] == DatasetRecordSourceType.CODE.value]
    assert len(code_sources) == 3

    explicit_records = [
        {
            "inputs": {"question": "Document-based question"},
            "expectations": {"answer": "From document"},
            "source": {
                "source_type": DatasetRecordSourceType.DOCUMENT.value,
                "source_data": {"source_id": "doc123", "page": 5},
            },
        }
    ]
    dataset.merge_records(explicit_records)

    df = dataset.to_df()
    doc_sources = df[df["source_type"] == DatasetRecordSourceType.DOCUMENT.value]
    assert len(doc_sources) == 1
    assert doc_sources.iloc[0]["source_id"] == "doc123"

    empty_exp_records = [{"inputs": {"question": "Has empty expectations"}, "expectations": {}}]
    dataset.merge_records(empty_exp_records)

    df = dataset.to_df()
    last_record = df.iloc[-1]
    assert last_record["source_type"] not in [
        DatasetRecordSourceType.HUMAN.value,
        DatasetRecordSourceType.CODE.value,
    ]

    explicit_trace = [
        {
            "inputs": {"question": "From trace"},
            "source": {
                "source_type": DatasetRecordSourceType.TRACE.value,
                "source_data": {"trace_id": "trace123"},
            },
        }
    ]
    dataset.merge_records(explicit_trace)

    df = dataset.to_df()
    trace_sources = df[df["source_type"] == DatasetRecordSourceType.TRACE.value]
    assert len(trace_sources) == 1, f"Expected 1 TRACE source, got {len(trace_sources)}"
    assert trace_sources.iloc[0]["source_id"] == "trace123"

    source_counts = df["source_type"].value_counts()
    assert source_counts.get(DatasetRecordSourceType.HUMAN.value, 0) == 2
    assert source_counts.get(DatasetRecordSourceType.CODE.value, 0) == 3
    assert source_counts.get(DatasetRecordSourceType.DOCUMENT.value, 0) == 1
    assert source_counts.get(DatasetRecordSourceType.TRACE.value, 0) == 1

    delete_dataset(dataset_id=dataset.dataset_id)


def test_trace_source_type_detection():
    exp = mlflow.create_experiment("test_trace_source_detection")

    trace_ids = []
    for i in range(3):
        with mlflow.start_run(experiment_id=exp):
            with mlflow.start_span(name=f"test_span_{i}") as span:
                span.set_inputs({"question": f"Question {i}", "context": f"Context {i}"})
                span.set_outputs({"answer": f"Answer {i}"})
                trace_ids.append(span.trace_id)

                if i < 2:
                    mlflow.log_expectation(
                        trace_id=span.trace_id,
                        name="quality",
                        value=0.8 + i * 0.05,
                        span_id=span.span_id,
                    )

    dataset = create_dataset(
        name="test_trace_sources", experiment_id=exp, tags={"test": "trace_source_detection"}
    )

    client = mlflow.MlflowClient()
    traces = [client.get_trace(tid) for tid in trace_ids]
    dataset.merge_records(traces)

    df = dataset.to_df()
    trace_sources = df[df["source_type"] == DatasetRecordSourceType.TRACE.value]
    assert len(trace_sources) == 3

    for idx, trace_id in enumerate(trace_ids):
        matching_records = df[df["source_id"] == trace_id]
        assert len(matching_records) == 1

    dataset2 = create_dataset(
        name="test_trace_sources_df", experiment_id=exp, tags={"test": "trace_source_df"}
    )

    traces_df = mlflow.search_traces(experiment_ids=[exp])
    assert not traces_df.empty
    dataset2.merge_records(traces_df)

    df2 = dataset2.to_df()
    trace_sources2 = df2[df2["source_type"] == DatasetRecordSourceType.TRACE.value]
    assert len(trace_sources2) == len(traces_df)

    dataset3 = create_dataset(
        name="test_trace_sources_list", experiment_id=exp, tags={"test": "trace_source_list"}
    )

    traces_list = mlflow.search_traces(experiment_ids=[exp], return_type="list")
    assert len(traces_list) > 0
    dataset3.merge_records(traces_list)

    df3 = dataset3.to_df()
    trace_sources3 = df3[df3["source_type"] == DatasetRecordSourceType.TRACE.value]
    assert len(trace_sources3) == len(traces_list)

    df_with_expectations = df[df["expectations"].apply(lambda x: bool(x) and len(x) > 0)]
    assert len(df_with_expectations) == 2

    delete_dataset(dataset_id=dataset.dataset_id)
    delete_dataset(dataset_id=dataset2.dataset_id)
    delete_dataset(dataset_id=dataset3.dataset_id)


def test_create_dataset_explicit_overrides_active_experiment(tracking_uri):
    active_exp = mlflow.create_experiment("active_exp")
    explicit_exp = mlflow.create_experiment("explicit_exp")

    mlflow.set_experiment(experiment_id=active_exp)

    dataset = create_dataset(name="test_explicit_override", experiment_id=explicit_exp)

    assert dataset.experiment_ids == [explicit_exp]

    from mlflow.tracking import fluent

    fluent._active_experiment_id = None


def test_create_dataset_none_uses_active_experiment(tracking_uri):
    exp_id = mlflow.create_experiment("test_none_experiment")
    mlflow.set_experiment(experiment_id=exp_id)

    dataset = create_dataset(name="test_none_exp", experiment_id=None)

    assert dataset.experiment_ids == [exp_id]

    from mlflow.tracking import fluent

    fluent._active_experiment_id = None


def test_create_dataset_empty_list_stays_empty(tracking_uri):
    exp_id = mlflow.create_experiment("test_empty_list")
    mlflow.set_experiment(experiment_id=exp_id)

    dataset = create_dataset(name="test_empty_list", experiment_id=[])

    assert dataset.experiment_ids == []

    from mlflow.tracking import fluent

    fluent._active_experiment_id = None


def test_search_datasets_filter_string_edge_cases(tracking_uri):
    exp_id = mlflow.create_experiment("test_filter_edge_cases")

    dataset = create_dataset(name="test_dataset", experiment_id=exp_id, tags={"test": "value"})

    with mock.patch("mlflow.tracking.client.MlflowClient.search_datasets") as mock_search:
        mock_search.return_value = mock.MagicMock(token=None, items=[dataset])

        search_datasets(experiment_ids=exp_id, filter_string=None)
        call_args = mock_search.call_args
        filter_arg = call_args.kwargs.get("filter_string")
        assert "created_time >=" in filter_arg

        mock_search.reset_mock()

        search_datasets(experiment_ids=exp_id, filter_string=[])
        call_args = mock_search.call_args
        filter_arg = call_args.kwargs.get("filter_string")
        assert "created_time >=" in filter_arg

        mock_search.reset_mock()

        search_datasets(experiment_ids=exp_id, filter_string="")
        call_args = mock_search.call_args
        filter_arg = call_args.kwargs.get("filter_string")
        assert "created_time >=" in filter_arg

        mock_search.reset_mock()

        search_datasets(experiment_ids=exp_id, filter_string='name = "test"')
        call_args = mock_search.call_args
        filter_arg = call_args.kwargs.get("filter_string")
        assert filter_arg == 'name = "test"'


def test_wrapper_type_is_actually_returned_not_entity(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_wrapper",
        experiment_id=experiments[0],
        tags={"test": "wrapper_check"},
    )

    assert isinstance(dataset, WrapperEvaluationDataset)
    assert not isinstance(dataset, EntityEvaluationDataset)
    assert hasattr(dataset, "_mlflow_dataset")
    assert dataset._mlflow_dataset is not None
    assert isinstance(dataset._mlflow_dataset, EntityEvaluationDataset)


def test_wrapper_delegates_all_properties_correctly(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_delegation",
        experiment_id=experiments[0],
        tags={"env": "test", "version": "1.0"},
    )

    assert dataset.name == "test_delegation"
    assert dataset.dataset_id.startswith("d-")
    assert dataset.tags["env"] == "test"
    assert dataset.tags["version"] == "1.0"
    assert experiments[0] in dataset.experiment_ids
    assert dataset.created_time > 0
    assert dataset.last_update_time > 0
    assert dataset.digest is not None
    assert hasattr(dataset, "source")
    assert dataset.source._get_source_type() == "mlflow_evaluation_dataset"


def test_get_and_search_return_wrapper_not_entity(tracking_uri, experiments):
    created = create_dataset(
        name="test_get_wrapper",
        experiment_id=experiments[0],
        tags={"test": "get"},
    )

    retrieved = get_dataset(dataset_id=created.dataset_id)
    assert isinstance(retrieved, WrapperEvaluationDataset)
    assert not isinstance(retrieved, EntityEvaluationDataset)
    assert retrieved.dataset_id == created.dataset_id
    assert retrieved.name == created.name

    results = search_datasets(
        experiment_ids=experiments[0],
        filter_string="name = 'test_get_wrapper'",
    )
    assert len(results) == 1
    assert isinstance(results[0], WrapperEvaluationDataset)
    assert not isinstance(results[0], EntityEvaluationDataset)


def test_wrapper_vs_direct_client_usage(tracking_uri, experiments):
    client = MlflowClient()

    entity_dataset = client.create_dataset(
        name="test_client_direct",
        experiment_id=experiments[0],
        tags={"direct": "client"},
    )
    assert isinstance(entity_dataset, EntityEvaluationDataset)
    assert not isinstance(entity_dataset, WrapperEvaluationDataset)

    wrapped_dataset = create_dataset(
        name="test_wrapped",
        experiment_id=experiments[0],
        tags={"wrapped": "fluent"},
    )
    assert isinstance(wrapped_dataset, WrapperEvaluationDataset)
    assert not isinstance(wrapped_dataset, EntityEvaluationDataset)
    assert wrapped_dataset._mlflow_dataset is not None

    wrapped_from_entity = WrapperEvaluationDataset(entity_dataset)
    assert wrapped_from_entity == entity_dataset


def test_wrapper_works_with_mlflow_log_input_integration(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_log_input",
        experiment_id=experiments[0],
    )

    records = [
        {
            "inputs": {"question": "Test question"},
            "expectations": {"answer": "Test answer"},
        }
    ]
    dataset.merge_records(records)

    with mlflow.start_run(experiment_id=experiments[0]) as run:
        mlflow.log_input(dataset, context="evaluation")

    run_data = mlflow.get_run(run.info.run_id)
    assert len(run_data.inputs.dataset_inputs) == 1
    dataset_input = run_data.inputs.dataset_inputs[0]
    assert dataset_input.dataset.name == "test_log_input"
    assert dataset_input.dataset.digest == dataset.digest


def test_wrapper_isinstance_checks_for_dataset_interfaces(tracking_uri, experiments):
    dataset = create_dataset(
        name="test_isinstance",
        experiment_id=experiments[0],
    )

    assert isinstance(dataset, Dataset)
    assert isinstance(dataset, PyFuncConvertibleDatasetMixin)
    assert isinstance(dataset, WrapperEvaluationDataset)
    assert not isinstance(dataset, EntityEvaluationDataset)
    assert isinstance(dataset, (WrapperEvaluationDataset, EntityEvaluationDataset))
