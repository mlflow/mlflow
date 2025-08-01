import sys
from unittest import mock

import pytest

from mlflow.entities.evaluation_dataset import EvaluationDataset as EntityEvaluationDataset
from mlflow.genai.datasets import (
    EvaluationDataset,
    create_evaluation_dataset,
    delete_evaluation_dataset,
    get_evaluation_dataset,
    search_evaluation_datasets,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_EVALUATION_DATASETS_MAX_RESULTS


@pytest.fixture
def mock_client():
    with mock.patch("mlflow.genai.datasets.MlflowClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        yield mock_client_instance


@pytest.fixture
def mock_databricks_environment():
    with mock.patch("mlflow.genai.datasets.is_in_databricks_runtime", return_value=True):
        yield


def test_create_evaluation_dataset(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
        source_type="HUMAN",
        source="manual",
    )
    mock_client.create_evaluation_dataset.return_value = expected_dataset

    result = create_evaluation_dataset(
        name="test_dataset",
        experiment_ids=["exp1", "exp2"],
        source_type="HUMAN",
        source="manual",
    )

    assert result == expected_dataset
    mock_client.create_evaluation_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_ids=["exp1", "exp2"],
        source_type="HUMAN",
        source="manual",
    )


def test_create_evaluation_dataset_single_experiment_id(mock_client):
    expected_dataset = EntityEvaluationDataset(dataset_id="test_id", name="test_dataset")
    mock_client.create_evaluation_dataset.return_value = expected_dataset

    result = create_evaluation_dataset(
        name="test_dataset",
        experiment_ids="exp1",
    )

    assert result == expected_dataset
    mock_client.create_evaluation_dataset.assert_called_once_with(
        name="test_dataset",
        experiment_ids=["exp1"],
        source_type=None,
        source=None,
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
            "catalog.schema.table", "exp1"
        )
        assert isinstance(result, EvaluationDataset)


def test_get_evaluation_dataset(mock_client):
    expected_dataset = EntityEvaluationDataset(
        dataset_id="test_id",
        name="test_dataset",
    )
    mock_client.get_evaluation_dataset.return_value = expected_dataset

    result = get_evaluation_dataset(dataset_id="test_id")

    assert result == expected_dataset
    mock_client.get_evaluation_dataset.assert_called_once_with("test_id")


def test_get_evaluation_dataset_missing_id():
    with pytest.raises(ValueError, match="Parameter 'dataset_id' is required in OSS environment"):
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
    with pytest.raises(ValueError, match="Parameter 'dataset_id' is required in OSS environment"):
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
        EntityEvaluationDataset(dataset_id="id1", name="dataset1"),
        EntityEvaluationDataset(dataset_id="id2", name="dataset2"),
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
    datasets = [EntityEvaluationDataset(dataset_id="id1", name="dataset1")]
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
    with mock.patch("mlflow.genai.datasets.is_in_databricks_runtime", return_value=True):
        with mock.patch.dict("sys.modules", {"databricks.agents.datasets": None}):
            with mock.patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="databricks-agents"):
                    create_evaluation_dataset(name="test", experiment_ids="exp1")
