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
    generator = EvaluationDemoGenerator()
    original_version = generator.version
    yield generator
    EvaluationDemoGenerator.version = original_version


@pytest.fixture
def traces_generator():
    return TracesDemoGenerator()


def test_generator_attributes(evaluation_generator):
    assert evaluation_generator.name == DemoFeature.EVALUATION
    assert evaluation_generator.version == 1


def test_data_exists_false_when_no_experiment(evaluation_generator):
    assert evaluation_generator._data_exists() is False


def test_data_exists_false_when_no_eval_runs(evaluation_generator, traces_generator):
    traces_generator.generate()
    assert evaluation_generator._data_exists() is False


def test_generate_creates_eval_runs(evaluation_generator):
    result = evaluation_generator.generate()
    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.EVALUATION
    assert len(result.entity_ids) == 2  # Two run IDs returned


def test_generate_creates_two_runs(evaluation_generator):
    evaluation_generator.generate()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.demo = 'true'",
    )
    assert len(runs) == 2


def test_data_exists_true_after_generate(evaluation_generator):
    evaluation_generator.generate()
    assert evaluation_generator._data_exists() is True


def test_delete_demo_removes_runs(evaluation_generator):
    evaluation_generator.generate()
    assert evaluation_generator._data_exists() is True
    evaluation_generator.delete_demo()
    assert evaluation_generator._data_exists() is False


def test_runs_have_demo_param(evaluation_generator):
    evaluation_generator.generate()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.demo = 'true'",
    )

    for run in runs:
        assert run.data.params.get("demo") == "true"


def test_runs_have_different_names(evaluation_generator):
    evaluation_generator.generate()

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.demo = 'true'",
    )

    run_names = {run.data.tags.get("mlflow.runName") for run in runs}
    assert BASELINE_PROFILE["name"] in run_names
    assert IMPROVED_PROFILE["name"] in run_names


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


def test_is_generated_checks_version(evaluation_generator):
    evaluation_generator.generate()
    evaluation_generator.store_version()

    assert evaluation_generator.is_generated() is True

    EvaluationDemoGenerator.version = 99
    fresh_generator = EvaluationDemoGenerator()
    assert fresh_generator.is_generated() is False
