import os
import sys

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import get_scorer, list_scorer_versions, scorer, update_scorer
from mlflow.tracking import MlflowClient


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
def experiment_id(client):
    experiment_id = client.create_experiment("test_scorer_experiment")
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def test_update_scorer_end_to_end(tracking_uri, experiment_id):
    @scorer
    def test_scorer(trace) -> bool:
        return True

    registered_scorer = test_scorer.register(
        experiment_id=experiment_id, name="test_scorer_for_update"
    )

    assert registered_scorer._sampling_config is not None
    assert registered_scorer._sampling_config.sample_rate == 0.0

    updated_scorer = update_scorer(
        name="test_scorer_for_update", experiment_id=experiment_id, sample_rate=0.5
    )

    assert updated_scorer._sampling_config.sample_rate == 0.5

    retrieved_scorer = get_scorer(name="test_scorer_for_update", experiment_id=experiment_id)
    assert retrieved_scorer._sampling_config.sample_rate == 0.5

    update_scorer(name="test_scorer_for_update", experiment_id=experiment_id, sample_rate=0.0)
    assert (
        get_scorer(
            name="test_scorer_for_update", experiment_id=experiment_id
        )._sampling_config.sample_rate
        == 0.0
    )

    update_scorer(name="test_scorer_for_update", experiment_id=experiment_id, sample_rate=1.0)
    assert (
        get_scorer(
            name="test_scorer_for_update", experiment_id=experiment_id
        )._sampling_config.sample_rate
        == 1.0
    )


def test_update_scorer_validation(tracking_uri, experiment_id):
    @scorer
    def validation_scorer(trace) -> bool:
        return True

    validation_scorer.register(experiment_id=experiment_id, name="validation_scorer")

    with pytest.raises(MlflowException, match="Invalid sample_rate.*Must be between 0.0 and 1.0"):
        update_scorer(name="validation_scorer", experiment_id=experiment_id, sample_rate=1.5)

    with pytest.raises(MlflowException, match="Invalid sample_rate.*Must be between 0.0 and 1.0"):
        update_scorer(name="validation_scorer", experiment_id=experiment_id, sample_rate=-0.1)

    with pytest.raises(MlflowException, match="Scorer with name 'nonexistent' not found"):
        update_scorer(name="nonexistent", experiment_id=experiment_id, sample_rate=0.5)


def test_update_scorer_with_multiple_versions(tracking_uri, experiment_id):
    @scorer
    def versioned_scorer_v1(trace) -> bool:
        return True

    versioned_scorer_v1.register(experiment_id=experiment_id, name="versioned_scorer")

    @scorer
    def versioned_scorer_v2(trace) -> bool:
        return len(trace.request_id) > 0 if hasattr(trace, "request_id") else False

    versioned_scorer_v2.register(experiment_id=experiment_id, name="versioned_scorer")

    updated_scorer = update_scorer(
        name="versioned_scorer", experiment_id=experiment_id, sample_rate=0.75
    )

    assert updated_scorer._sampling_config.sample_rate == 0.75

    latest = get_scorer(name="versioned_scorer", experiment_id=experiment_id)
    assert latest._sampling_config.sample_rate == 0.75

    versions = list_scorer_versions(name="versioned_scorer", experiment_id=experiment_id)
    assert len(versions) == 2

    _, v1_num = versions[0]
    v2_retrieved, v2_num = versions[1]

    assert v1_num == 1
    assert v2_num == 2
    assert v2_retrieved._sampling_config.sample_rate == 0.75


def test_update_scorer_none_preserves_value(tracking_uri, experiment_id):
    @scorer
    def none_test_scorer(trace) -> bool:
        return True

    none_test_scorer.register(experiment_id=experiment_id, name="none_test_scorer")

    update_scorer(name="none_test_scorer", experiment_id=experiment_id, sample_rate=0.7)

    before_none = get_scorer(name="none_test_scorer", experiment_id=experiment_id)
    assert before_none._sampling_config.sample_rate == 0.7

    updated_with_none = update_scorer(
        name="none_test_scorer", experiment_id=experiment_id, sample_rate=None
    )

    assert updated_with_none._sampling_config.sample_rate == 0.7

    after_none = get_scorer(name="none_test_scorer", experiment_id=experiment_id)
    assert after_none._sampling_config.sample_rate == 0.7
