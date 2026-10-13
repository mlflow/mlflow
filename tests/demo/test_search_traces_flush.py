from unittest import mock

from mlflow import set_experiment
from mlflow.demo.base import DEMO_EXPERIMENT_NAME
from mlflow.demo.generators.evaluation import EvaluationDemoGenerator
from mlflow.demo.generators.issues import IssuesDemoGenerator
from mlflow.demo.generators.traces import TracesDemoGenerator


def test_evaluation_fetch_demo_traces_passes_flush():
    with mock.patch(
        "mlflow.demo.generators.evaluation.mlflow.search_traces",
        return_value=[],
    ) as mock_search:
        EvaluationDemoGenerator()._fetch_demo_traces(experiment_id="0", version="v2", session=True)

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs.get("flush") is True


def test_issues_generate_passes_flush():
    set_experiment(DEMO_EXPERIMENT_NAME)
    with mock.patch(
        "mlflow.demo.generators.issues.mlflow.search_traces",
        return_value=[],
    ) as mock_search:
        IssuesDemoGenerator().generate()

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs.get("flush") is True


def test_traces_data_exists_passes_flush():
    set_experiment(DEMO_EXPERIMENT_NAME)
    with mock.patch(
        "mlflow.demo.generators.traces.mlflow.search_traces",
        return_value=[],
    ) as mock_search:
        TracesDemoGenerator()._data_exists()

    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs.get("flush") is True
