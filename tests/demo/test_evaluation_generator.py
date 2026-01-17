import pytest

import mlflow
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.data import ALL_DEMO_TRACES
from mlflow.demo.generators.evaluation import (
    BASELINE_PROFILE,
    IMPROVED_PROFILE,
    EvaluationDemoGenerator,
)
from mlflow.demo.generators.traces import TracesDemoGenerator


@pytest.fixture
def evaluation_generator():
    return EvaluationDemoGenerator()


@pytest.fixture
def traces_generator():
    return TracesDemoGenerator()


def test_generator_attributes(evaluation_generator):
    assert evaluation_generator.name == DemoFeature.EVALUATION
    assert evaluation_generator.version == 1


def test_data_exists_false_when_no_experiment(evaluation_generator, tracking_uri):
    assert evaluation_generator._data_exists() is False


def test_data_exists_false_when_no_eval_runs(evaluation_generator, traces_generator, tracking_uri):
    traces_generator.generate()
    assert evaluation_generator._data_exists() is False


def test_generate_creates_eval_runs(evaluation_generator, tracking_uri):
    result = evaluation_generator.generate()
    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.EVALUATION
    assert any("eval_runs" in e for e in result.entity_ids)


def test_generate_creates_two_runs(evaluation_generator, tracking_uri):
    evaluation_generator.generate()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.demo = 'true'",
    )
    assert len(runs) == 2


def test_data_exists_true_after_generate(evaluation_generator, tracking_uri):
    evaluation_generator.generate()
    assert evaluation_generator._data_exists() is True


def test_delete_demo_removes_runs(evaluation_generator, tracking_uri):
    evaluation_generator.generate()
    assert evaluation_generator._data_exists() is True
    evaluation_generator.delete_demo()
    assert evaluation_generator._data_exists() is False


def test_runs_have_demo_param(evaluation_generator, tracking_uri):
    evaluation_generator.generate()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.demo = 'true'",
    )

    for run in runs:
        assert run.data.params.get("demo") == "true"
        assert run.data.params.get("scorer_version") is not None
        assert run.data.params.get("description") is not None


def test_runs_have_different_versions(evaluation_generator, tracking_uri):
    evaluation_generator.generate()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.demo = 'true'",
    )

    versions = {run.data.params.get("scorer_version") for run in runs}
    assert BASELINE_PROFILE["version"] in versions
    assert IMPROVED_PROFILE["version"] in versions


def test_demo_traces_have_responses():
    assert len(ALL_DEMO_TRACES) > 0
    for trace in ALL_DEMO_TRACES:
        assert isinstance(trace.query, str)
        assert isinstance(trace.v1_response, str)
        assert isinstance(trace.v2_response, str)
        assert isinstance(trace.expected_response, str)
        assert len(trace.v1_response) > 20
        assert len(trace.v2_response) > 20
        assert len(trace.expected_response) > 20


def test_is_generated_checks_version(evaluation_generator, tracking_uri):
    evaluation_generator.generate()
    evaluation_generator.store_version()

    assert evaluation_generator.is_generated() is True

    EvaluationDemoGenerator.version = 99
    fresh_generator = EvaluationDemoGenerator()
    assert fresh_generator.is_generated() is False

    EvaluationDemoGenerator.version = 1
