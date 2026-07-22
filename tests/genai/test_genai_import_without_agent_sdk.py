import json
from unittest import mock
from unittest.mock import patch

import pytest

from mlflow.genai.datasets import create_dataset, delete_dataset, get_dataset
from mlflow.genai.scorers import (
    Guidelines,
    delete_scorer,
    get_scorer,
    list_scorer_versions,
    list_scorers,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.registry import DatabricksStore


# Test `mlflow.genai` namespace
def test_mlflow_genai_star_import_succeeds():
    import mlflow.genai  # noqa: F401


def test_namespaced_import_raises_when_agents_not_installed():
    # Ensure that databricks-agents methods renamespaced under mlflow.genai raise an
    # ImportError when the databricks-agents package is not installed.
    import mlflow.genai

    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            mlflow.genai.create_dataset("test_schema")

        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            mlflow.genai.get_dataset("test_schema")

        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            mlflow.genai.delete_dataset("test_schema")


# Test `mlflow.genai.datasets` namespace
def test_mlflow_genai_datasets_star_import_succeeds():
    import mlflow.genai.datasets  # noqa: F401


def test_create_dataset_raises_when_agents_not_installed():
    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            create_dataset("test_dataset")


def test_get_dataset_raises_when_agents_not_installed():
    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            get_dataset("test_dataset")


def test_delete_dataset_raises_when_agents_not_installed():
    # Mock to simulate Databricks environment without databricks-agents installed
    with patch("mlflow.genai.datasets.is_databricks_uri", return_value=True):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            delete_dataset("test_dataset")


class MockScorer(Scorer):
    """Mock scorer for testing purposes."""

    name: str = "mock_scorer"

    def __call__(self, *, outputs=None, **kwargs):
        return {"score": 1.0}


def test_list_scorers_raises_when_agents_not_installed():
    with patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
    ):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            list_scorers(experiment_id="test_experiment")


def test_get_scorer_raises_when_agents_not_installed():
    with patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
    ):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            get_scorer(name="test_scorer", experiment_id="test_experiment")


def test_delete_scorer_raises_when_agents_not_installed():
    with patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
    ):
        with pytest.raises(ImportError, match="The `databricks-agents` package is required"):
            delete_scorer(experiment_id="test_experiment", name="test_scorer")


def _mock_response(payload):
    response = mock.MagicMock()
    response.status_code = 200
    response.text = json.dumps(payload)
    response.json.return_value = payload
    return response


def _scheduled_scorers_response(configs):
    return _mock_response({"scheduled_scorers": {"scorers": configs}})


def _scorer_config(name="test_scorer"):
    scorer = Guidelines(name=name, guidelines=["Be helpful"], model="databricks:/judge")
    return {
        "name": name,
        "serialized_scorer": json.dumps(scorer.model_dump()),
        "builtin": {"name": name},
        "sample_rate": 0.0,
        "scorer_version": 1,
    }


def _instructions_judge_config(name="instructions_judge_scorer"):
    return {
        "name": name,
        "serialized_scorer": json.dumps({
            "name": name,
            "aggregations": None,
            "description": None,
            "is_session_level_scorer": False,
            "mlflow_version": "3.0.0",
            "serialization_version": 1,
            "builtin_scorer_class": None,
            "builtin_scorer_pydantic_data": None,
            "call_source": None,
            "call_signature": None,
            "original_func_name": None,
            "instructions_judge_pydantic_data": {
                "instructions": "Assess {{ outputs }}.",
                "model": "databricks",
            },
            "memory_augmented_judge_data": None,
            "third_party_scorer_data": None,
        }),
        "custom": {},
        "sample_rate": 0.0,
        "scorer_version": 1,
    }


@pytest.fixture
def scorer_http():
    with (
        patch(
            "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
        ),
        patch("mlflow.genai.scorers.registry.get_databricks_host_creds", return_value="creds"),
        patch("mlflow.genai.scorers.registry.http_request") as mock_http,
    ):
        yield mock_http


def test_versioned_get_with_databricks_instructions_judge_does_not_require_agents_sdk(
    scorer_http,
):
    config = _instructions_judge_config()
    scorer_key = DatabricksStore._scorer_resource_key("instructions_judge_scorer")
    version_config = {
        **config,
        "name": f"experiments/test_experiment/scorers/{scorer_key}/versions/1",
        "display_name": "instructions_judge_scorer",
    }
    scorer_http.side_effect = [
        _mock_response(version_config),
        _scheduled_scorers_response([config]),
    ]

    scorer = get_scorer(
        name="instructions_judge_scorer",
        experiment_id="test_experiment",
        version=1,
    )

    assert scorer.name == "instructions_judge_scorer"
    assert scorer.model == "databricks"


def test_versioned_scorer_operations_do_not_require_agents_sdk(scorer_http):
    config = _scorer_config()
    scorer_key = DatabricksStore._scorer_resource_key("test_scorer")
    version_config = {
        "name": f"experiments/test_experiment/scorers/{scorer_key}/versions/1",
        "display_name": "test_scorer",
        "scorer_version": 1,
        "serialized_scorer": config["serialized_scorer"],
        "builtin": {"name": "test_scorer"},
    }
    scorer_http.side_effect = [
        _mock_response(version_config),
        _scheduled_scorers_response([config]),
        _mock_response({"scorer_versions": [version_config]}),
        _scheduled_scorers_response([config]),
        _mock_response({}),
    ]

    exact = get_scorer(name="test_scorer", experiment_id="test_experiment", version=1)
    versions = list_scorer_versions(name="test_scorer", experiment_id="test_experiment")
    delete_scorer(name="test_scorer", experiment_id="test_experiment", version=1)

    assert exact.name == "test_scorer"
    assert [version for _, version in versions] == [1]
    assert scorer_http.call_args_list[-1].kwargs["method"] == "DELETE"
