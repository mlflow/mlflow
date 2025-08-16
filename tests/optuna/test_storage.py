import os
import random
import tempfile
import time
from datetime import datetime
from time import sleep
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.optuna.storage import MlflowStorage

ALL_STATES = list(TrialState)

EXAMPLE_ATTRS: dict[str, Any] = {
    "dataset": "MNIST",
    "none": None,
    "json_serializable": {"baseline_score": 0.001, "tags": ["image", "classification"]},
}


def _setup_studies(
    storage: BaseStorage,
    n_study: int,
    n_trial: int,
    seed: int,
    direction: StudyDirection = None,
) -> tuple[dict[int, FrozenStudy], dict[int, dict[int, FrozenTrial]]]:
    generator = random.Random(seed)
    study_id_to_frozen_study: dict[int, FrozenStudy] = {}
    study_id_to_trials: dict[int, dict[int, FrozenTrial]] = {}
    for i in range(n_study):
        study_name = "test-study-name-{}".format(i)
        if direction is None:
            direction = generator.choice([StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
        study_id = storage.create_new_study(directions=(direction,), study_name=study_name)
        storage.set_study_user_attr(study_id, "u", i)
        storage.set_study_system_attr(study_id, "s", i)
        trials = {}
        for j in range(n_trial):
            trial = _generate_trial(generator)
            trial.number = j
            trial._trial_id = storage.create_new_trial(study_id, trial)
            trials[trial._trial_id] = trial
        study_id_to_trials[study_id] = trials
        study_id_to_frozen_study[study_id] = FrozenStudy(
            study_name=study_name,
            direction=direction,
            user_attrs={"u": i},
            system_attrs={"s": i},
            study_id=study_id,
        )
    return study_id_to_frozen_study, study_id_to_trials


def _generate_trial(generator: random.Random) -> FrozenTrial:
    example_params = {
        "paramA": (generator.uniform(0, 1), FloatDistribution(0, 1)),
        "paramB": (generator.uniform(1, 2), FloatDistribution(1, 2, log=True)),
        "paramC": (
            generator.choice(["CatA", "CatB", "CatC"]),
            CategoricalDistribution(("CatA", "CatB", "CatC")),
        ),
        "paramD": (generator.uniform(-3, 0), FloatDistribution(-3, 0)),
        "paramE": (generator.choice([0.1, 0.2]), CategoricalDistribution((0.1, 0.2))),
    }
    example_attrs = {
        "attrA": "valueA",
        "attrB": 1,
        "attrC": None,
        "attrD": {"baseline_score": 0.001, "tags": ["image", "classification"]},
    }
    state = generator.choice(ALL_STATES)
    params = {}
    distributions = {}
    user_attrs = {}
    system_attrs: dict[str, Any] = {}
    intermediate_values = {}
    for key, (value, dist) in example_params.items():
        if generator.choice([True, False]):
            params[key] = value
            distributions[key] = dist
    for key, value in example_attrs.items():
        if generator.choice([True, False]):
            user_attrs["usr_" + key] = value
        if generator.choice([True, False]):
            system_attrs["sys_" + key] = value
    for i in range(generator.randint(4, 10)):
        if generator.choice([True, False]):
            intermediate_values[i] = generator.uniform(-10, 10)
    return FrozenTrial(
        number=0,  # dummy
        state=state,
        value=generator.uniform(-10, 10),
        datetime_start=datetime.now(),
        datetime_complete=datetime.now() if state.is_finished() else None,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        intermediate_values=intermediate_values,
        trial_id=0,  # dummy
    )


@pytest.fixture
def setup_storage():
    tempdir = tempfile.mkdtemp(prefix="optuna_tests_", dir="/tmp")
    mlflow_uri = "file:" + os.path.join(tempdir, "mlflow")
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_id = mlflow.create_experiment(name="optuna_mlflow_test")
    storage = MlflowStorage(
        experiment_id=experiment_id, batch_flush_interval=1.0, batch_size_threshold=5
    )
    storage._flush_thread = MagicMock()
    yield storage
    mlflow.delete_experiment(experiment_id)


def test_queue_batch_operation_creates_new_queue_for_new_run(setup_storage):
    storage = setup_storage
    run_id = "test-run-id"
    test_metric = Metric("test_metric", 1.0, int(time.time() * 1000), 0)

    # Call the method with a new run_id
    storage._queue_batch_operation(run_id, metrics=[test_metric])

    # Check that a new queue was created for this run_id
    assert run_id in storage._batch_queue
    assert len(storage._batch_queue[run_id]["metrics"]) == 1
    assert storage._batch_queue[run_id]["metrics"][0] == test_metric
    assert len(storage._batch_queue[run_id]["params"]) == 0
    assert len(storage._batch_queue[run_id]["tags"]) == 0


def test_queue_batch_operation_appends_to_existing_queue(setup_storage):
    storage = setup_storage
    run_id = "test-run-id"

    # Setup existing queue
    storage._batch_queue[run_id] = {"metrics": [], "params": [], "tags": []}

    # Add metrics, params, and tags
    test_metric = Metric("test_metric", 1.0, int(time.time() * 1000), 0)
    test_param = Param("test_param", "value")
    test_tag = RunTag("test_tag", "value")

    # Queue each type separately
    storage._queue_batch_operation(run_id, metrics=[test_metric])
    storage._queue_batch_operation(run_id, params=[test_param])
    storage._queue_batch_operation(run_id, tags=[test_tag])

    # Check all were added correctly
    assert len(storage._batch_queue[run_id]["metrics"]) == 1
    assert len(storage._batch_queue[run_id]["params"]) == 1
    assert len(storage._batch_queue[run_id]["tags"]) == 1


def test_flush_batch_sends_data_to_mlflow(setup_storage):
    """Test that _flush_batch properly sends data to MLflow client."""
    with patch("mlflow.optuna.storage.MlflowClient") as mock_client:
        storage = setup_storage
        run_id = "test-run-id"
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage._mlflow_client = mock_client

        # Setup batch data
        metrics = [Metric("test_metric", 1.0, int(time.time() * 1000), 0)]
        params = [Param("test_param", "value")]
        tags = [RunTag("test_tag", "value")]

        storage._batch_queue[run_id] = {"metrics": metrics[:], "params": params[:], "tags": tags[:]}

        # Call _flush_batch
        storage._flush_batch(run_id)

        # Verify MLflow client was called with the correct data
        storage._mlflow_client.log_batch.assert_called_once_with(
            run_id, metrics=metrics, params=params, tags=tags
        )

        # Verify batch was cleared
        assert len(storage._batch_queue[run_id]["metrics"]) == 0
        assert len(storage._batch_queue[run_id]["params"]) == 0
        assert len(storage._batch_queue[run_id]["tags"]) == 0


def test_flush_batch_does_nothing_for_empty_batch(setup_storage):
    """Test that _flush_batch does nothing when batch is empty."""
    with patch("mlflow.optuna.storage.MlflowClient") as mock_client:
        storage = setup_storage
        run_id = "test-run-id"
        # Create a mock instance that will be returned when MlflowClient is instantiated
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage._mlflow_client = mock_client

        # Setup empty batch
        storage._batch_queue[run_id] = {"metrics": [], "params": [], "tags": []}

        # Call _flush_batch
        storage._flush_batch(run_id)

        # Verify MLflow client was not called
        storage._mlflow_client.log_batch.assert_not_called()


def test_flush_batch_handles_nonexistent_run(setup_storage):
    """Test that _flush_batch handles nonexistent run gracefully."""
    with patch("mlflow.optuna.storage.MlflowClient") as mock_client:
        storage = setup_storage
        run_id = "nonexistent-run"
        # Create a mock instance that will be returned when MlflowClient is instantiated
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        storage._mlflow_client = mock_client

        # Call _flush_batch for a run that doesn't exist in the queue
        storage._flush_batch(run_id)

        # Verify MLflow client was not called
        storage._mlflow_client.log_batch.assert_not_called()


def test_flush_all_batches_flushes_all_runs(setup_storage):
    """Test that flush_all_batches flushes all pending runs."""
    storage = setup_storage
    # Setup multiple runs with data
    run_ids = ["run1", "run2", "run3"]

    for run_id in run_ids:
        storage._batch_queue[run_id] = {
            "metrics": [Metric(f"m_{run_id}", 1.0, int(time.time() * 1000), 0)],
            "params": [Param(f"p_{run_id}", "value")],
            "tags": [RunTag(f"t_{run_id}", "value")],
        }

    # Create a spy on _flush_batch
    with patch.object(storage, "_flush_batch") as mock_flush:
        # Call flush_all_batches
        storage.flush_all_batches()

        # Check that _flush_batch was called for each run
        expected_calls = [call(run_id) for run_id in run_ids]
        mock_flush.assert_has_calls(expected_calls, any_order=True)
        assert mock_flush.call_count == len(run_ids)


def test_flush_all_batches_handles_empty_queue(setup_storage):
    """Test that flush_all_batches works with empty queue."""
    storage = setup_storage
    # Ensure batch queue is empty
    storage._batch_queue = {}

    # Create a spy on _flush_batch
    with patch.object(storage, "_flush_batch") as mock_flush:
        # Call flush_all_batches
        storage.flush_all_batches()

        # Check that _flush_batch was not called
        mock_flush.assert_not_called()


def test_create_new_study(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    frozen_studies = storage.get_all_studies()
    assert len(frozen_studies) == 1
    assert frozen_studies[0]._study_id == study_id
    assert frozen_studies[0].study_name.startswith(DEFAULT_STUDY_NAME_PREFIX)

    study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    # Study id must be unique.
    assert study_id != study_id2
    frozen_studies = storage.get_all_studies()
    assert len(frozen_studies) == 2
    assert {s._study_id for s in frozen_studies} == {study_id, study_id2}
    assert all(s.study_name.startswith(DEFAULT_STUDY_NAME_PREFIX) for s in frozen_studies)


def test_delete_study(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    storage.create_new_trial(study_id)
    trials = storage.get_all_trials(study_id)
    assert len(trials) == 1

    with pytest.raises(mlflow.exceptions.MlflowException, match="Run .* not found"):
        # Deletion of non-existent study.
        storage.delete_study(study_id + "1")

    storage.delete_study(study_id)


def test_get_study_id_from_name_and_get_study_name_from_id(setup_storage):
    storage = setup_storage
    study_name = "test_optuna_mlflow_study"
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE], study_name=study_name)

    # Test existing study.
    assert storage.get_study_name_from_id(study_id) == study_name
    assert storage.get_study_id_from_name(study_name) == study_id

    # Test not existing study.
    with pytest.raises(Exception, match="Study dummy-name not found"):
        storage.get_study_id_from_name("dummy-name")

    with pytest.raises(mlflow.exceptions.MlflowException, match="Run .* not found"):
        storage.get_study_name_from_id(study_id + "1")


def test_set_and_get_study_directions(setup_storage):
    storage = setup_storage
    for target in [
        (StudyDirection.MINIMIZE,),
        (StudyDirection.MAXIMIZE,),
        (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE),
        (StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE),
        [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE],
        [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE],
    ]:
        study_id = storage.create_new_study(directions=target)

        def check_get() -> None:
            got_directions = storage.get_study_directions(study_id)

            assert got_directions == list(target), (
                "Direction of a study should be a tuple of `StudyDirection` objects."
            )

        # Test setting value.
        check_get()

        # Test non-existent study.
        non_existent_study_id = study_id + "1"

        with pytest.raises(mlflow.exceptions.MlflowException, match="Run .* not found"):
            storage.get_study_directions(non_existent_study_id)


def test_set_and_get_study_user_attrs(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

    def check_set_and_get(key: str, value: Any) -> None:
        storage.set_study_user_attr(study_id, key, value)
        assert storage.get_study_user_attrs(study_id)[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(key, value)
    assert storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get("dataset", "ImageNet")

    # Non-existent study id.
    non_existent_study_id = study_id + "1"
    with pytest.raises(mlflow.exceptions.MlflowException, match="Run .* not found"):
        storage.get_study_user_attrs(non_existent_study_id)

    # Non-existent study id.
    with pytest.raises(mlflow.exceptions.MlflowException, match="Run .* not found"):
        storage.set_study_user_attr(non_existent_study_id, "key", "value")


def test_set_and_get_study_system_attrs(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

    def check_set_and_get(key: str, value: Any) -> None:
        storage.set_study_system_attr(study_id, key, value)
        assert storage.get_study_system_attrs(study_id)[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(key, value)
    assert storage.get_study_system_attrs(study_id) == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get("dataset", "ImageNet")


def test_create_new_trial(setup_storage):
    storage = setup_storage

    def _check_trials(
        trials: list[FrozenTrial],
        idx: int,
        trial_id: int,
        time_before_creation: datetime,
        time_after_creation: datetime,
    ) -> None:
        assert len(trials) == idx + 1
        assert len({t._trial_id for t in trials}) == idx + 1
        assert trial_id in {t._trial_id for t in trials}
        assert {t.number for t in trials} == set(range(idx + 1))
        assert all(t.state == TrialState.RUNNING for t in trials)
        assert all(t.params == {} for t in trials)
        assert all(t.intermediate_values == {} for t in trials)
        assert all(t.user_attrs == {} for t in trials)
        assert all(t.system_attrs == {} for t in trials)
        assert all(
            t.datetime_start < time_before_creation
            for t in trials
            if t._trial_id != trial_id and t.datetime_start is not None
        )
        assert all(
            time_before_creation < t.datetime_start < time_after_creation
            for t in trials
            if t._trial_id == trial_id and t.datetime_start is not None
        )
        assert all(t.value is None for t in trials)

    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    n_trial_in_study = 3
    for i in range(n_trial_in_study):
        time_before_creation = datetime.now()
        sleep(0.001)  # Sleep 1ms to avoid faulty assertion on Windows OS.
        trial_id = storage.create_new_trial(study_id)
        sleep(0.001)
        time_after_creation = datetime.now()

        trials = storage.get_all_trials(study_id)
        _check_trials(trials, i, trial_id, time_before_creation, time_after_creation)


def test_create_new_trial_with_template_trial(setup_storage):
    storage = setup_storage
    start_time = datetime.now()
    end_time = datetime.now()
    template_trial = FrozenTrial(
        state=TrialState.COMPLETE,
        value=10000,
        datetime_start=start_time,
        datetime_complete=end_time,
        params={"x": 0.5},
        distributions={"x": FloatDistribution(0, 1)},
        user_attrs={"foo": "bar"},
        system_attrs={"baz": 123},
        intermediate_values={1: 10, 2: 100, 3: 1000},
        number=55,  # This entry is ignored.
        trial_id=-1,  # dummy value (unused).
    )

    def _check_trials(trials: list[FrozenTrial], idx: int, trial_id: int) -> None:
        assert len(trials) == idx + 1
        assert len({t._trial_id for t in trials}) == idx + 1
        assert trial_id in {t._trial_id for t in trials}
        assert {t.number for t in trials} == set(range(idx + 1))
        assert all(t.state == template_trial.state for t in trials)

        assert all(t.params == template_trial.params for t in trials)
        assert all(t.distributions == template_trial.distributions for t in trials)
        assert all(t.intermediate_values == template_trial.intermediate_values for t in trials)
        assert all(t.user_attrs == template_trial.user_attrs for t in trials)
        assert all(t.system_attrs == template_trial.system_attrs for t in trials)
        # assert all(t.datetime_start == template_trial.datetime_start for t in trials)
        # assert all(t.datetime_complete == template_trial.datetime_complete for t in trials)
        assert all(t.value == template_trial.value for t in trials)

    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    n_trial_in_study = 3
    for i in range(n_trial_in_study):
        trial_id = storage.create_new_trial(study_id, template_trial=template_trial)
        trials = storage.get_all_trials(study_id)
        _check_trials(trials, i, trial_id)


def test_get_trial_number_from_id(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)
    assert storage.get_trial_number_from_id(trial_id) == 0

    trial_id = storage.create_new_trial(study_id)
    assert storage.get_trial_number_from_id(trial_id) == 1


def test_set_trial_state_values_for_state(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_ids = [storage.create_new_trial(study_id) for _ in ALL_STATES]

    for trial_id, state in zip(trial_ids, ALL_STATES):
        if state == TrialState.WAITING:
            continue
        assert storage.get_trial(trial_id).state == TrialState.RUNNING
        datetime_start_prev = storage.get_trial(trial_id).datetime_start
        storage.set_trial_state_values(
            trial_id, state=state, values=(0.0,) if state.is_finished() else None
        )
        assert storage.get_trial(trial_id).state == state
        # Repeated state changes to RUNNING should not trigger further datetime_start changes.
        if state == TrialState.RUNNING:
            assert storage.get_trial(trial_id).datetime_start == datetime_start_prev
        if state.is_finished():
            assert storage.get_trial(trial_id).datetime_complete is not None
        else:
            assert storage.get_trial(trial_id).datetime_complete is None


def test_get_trial_param_and_get_trial_params(setup_storage):
    storage = setup_storage
    _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=1)

    for _, trial_id_to_trial in study_to_trials.items():
        for trial_id, expected_trial in trial_id_to_trial.items():
            assert storage.get_trial_params(trial_id) == expected_trial.params
            for key in expected_trial.params.keys():
                assert storage.get_trial_param(trial_id, key) == expected_trial.distributions[
                    key
                ].to_internal_repr(expected_trial.params[key])


def test_set_trial_param(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id_1 = storage.create_new_trial(study_id)
    trial_id_2 = storage.create_new_trial(study_id)

    # Setup distributions.
    distribution_x = FloatDistribution(low=1.0, high=2.0)
    distribution_y_1 = CategoricalDistribution(choices=("Shibuya", "Ebisu", "Meguro"))
    distribution_z = FloatDistribution(low=1.0, high=100.0, log=True)

    # Set new params.
    storage.set_trial_param(trial_id_1, "x", 0.5, distribution_x)
    storage.set_trial_param(trial_id_1, "y", 2, distribution_y_1)
    assert storage.get_trial_param(trial_id_1, "x") == 0.5
    assert storage.get_trial_param(trial_id_1, "y") == 2
    # Check set_param breaks neither get_trial nor get_trial_params.
    assert storage.get_trial(trial_id_1).params == {"x": 0.5, "y": "Meguro"}
    assert storage.get_trial_params(trial_id_1) == {"x": 0.5, "y": "Meguro"}

    # Set params to another trial.
    storage.set_trial_param(trial_id_2, "x", 0.3, distribution_x)
    storage.set_trial_param(trial_id_2, "z", 0.1, distribution_z)
    assert storage.get_trial_param(trial_id_2, "x") == 0.3
    assert storage.get_trial_param(trial_id_2, "z") == 0.1
    assert storage.get_trial(trial_id_2).params == {"x": 0.3, "z": 0.1}
    assert storage.get_trial_params(trial_id_2) == {"x": 0.3, "z": 0.1}

    storage.set_trial_state_values(trial_id_2, state=TrialState.COMPLETE)
    # Cannot assign params to finished trial.
    with pytest.raises(RuntimeError, match="Trial#1 has already finished and can not be updated."):
        storage.set_trial_param(trial_id_2, "y", 2, distribution_y_1)
    # Check the previous call does not change the params.
    with pytest.raises(KeyError, match="'param_internal_val_y'"):
        storage.get_trial_param(trial_id_2, "y")
    # State should be checked prior to distribution compatibility.
    with pytest.raises(RuntimeError, match="Trial#1 has already finished and can not be updated."):
        storage.set_trial_param(trial_id_2, "y", 0.4, distribution_z)


def test_set_trial_state_values_for_values(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id_1 = storage.create_new_trial(study_id)
    trial_id_2 = storage.create_new_trial(study_id)
    trial_id_3 = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )
    trial_id_4 = storage.create_new_trial(study_id)
    trial_id_5 = storage.create_new_trial(study_id)

    # Test setting new value.
    storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE, values=(0.5,))
    storage.set_trial_state_values(trial_id_3, state=TrialState.COMPLETE, values=(float("inf"),))
    storage.set_trial_state_values(trial_id_4, state=TrialState.WAITING, values=(0.1, 0.2, 0.3))
    storage.set_trial_state_values(trial_id_5, state=TrialState.WAITING, values=[0.1, 0.2, 0.3])

    assert storage.get_trial(trial_id_1).value == 0.5
    assert storage.get_trial(trial_id_2).value is None
    assert storage.get_trial(trial_id_3).value == float("inf")
    assert storage.get_trial(trial_id_4).values == [0.1, 0.2, 0.3]
    assert storage.get_trial(trial_id_5).values == [0.1, 0.2, 0.3]


def test_set_trial_intermediate_value(setup_storage):
    storage = setup_storage
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id_1 = storage.create_new_trial(study_id)
    trial_id_2 = storage.create_new_trial(study_id)
    trial_id_3 = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )
    trial_id_4 = storage.create_new_trial(study_id)

    # Test setting new values.
    storage.set_trial_intermediate_value(trial_id_1, 0, 0.3)
    storage.set_trial_intermediate_value(trial_id_1, 2, 0.4)
    storage.set_trial_intermediate_value(trial_id_3, 0, 0.1)
    storage.set_trial_intermediate_value(trial_id_3, 1, 0.4)
    storage.set_trial_intermediate_value(trial_id_3, 2, 0.5)
    storage.set_trial_intermediate_value(trial_id_3, 3, float("inf"))
    storage.set_trial_intermediate_value(trial_id_4, 0, float("nan"))

    assert storage.get_trial(trial_id_1).intermediate_values == {0: 0.3, 2: 0.4}
    assert storage.get_trial(trial_id_2).intermediate_values == {}
    assert storage.get_trial(trial_id_3).intermediate_values == {
        0: 0.1,
        1: 0.4,
        2: 0.5,
        3: float("inf"),
    }
    assert np.isnan(storage.get_trial(trial_id_4).intermediate_values[0])

    # Test setting existing step.
    storage.set_trial_intermediate_value(trial_id_1, 0, 0.2)
    assert storage.get_trial(trial_id_1).intermediate_values == {0: 0.2, 2: 0.4}


def test_get_trial_user_attrs(setup_storage):
    storage = setup_storage
    _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=10)
    assert all(
        storage.get_trial_user_attrs(trial_id) == trial.user_attrs
        for trials in study_to_trials.values()
        for trial_id, trial in trials.items()
    )


def test_set_trial_user_attr(setup_storage):
    storage = setup_storage
    trial_id_1 = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )

    def check_set_and_get(trial_id: int, key: str, value: Any) -> None:
        storage.set_trial_user_attr(trial_id, key, value)
        assert storage.get_trial(trial_id).user_attrs[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(trial_id_1, key, value)
    assert storage.get_trial(trial_id_1).user_attrs == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get(trial_id_1, "dataset", "ImageNet")


def test_set_trial_system_attr(setup_storage):
    storage = setup_storage
    trial_id_1 = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )

    def check_set_and_get(trial_id: int, key: str, value: Any) -> None:
        storage.set_trial_system_attr(trial_id, key, value)
        assert storage.get_trial_system_attrs(trial_id)[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(trial_id_1, key, value)
    system_attrs = storage.get_trial(trial_id_1).system_attrs
    assert system_attrs == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get(trial_id_1, "dataset", "ImageNet")


def test_get_n_trials(setup_storage):
    storage = setup_storage
    study_id_to_frozen_studies, _ = _setup_studies(storage, n_study=2, n_trial=7, seed=50)
    for study_id in study_id_to_frozen_studies:
        assert storage.get_n_trials(study_id) == 7


def test_study_exists_method(setup_storage):
    storage = setup_storage

    # Test non-existent study
    assert not storage.get_study_id_by_name_if_exists("non-existent-study")

    # Create a study
    storage.create_new_study([StudyDirection.MINIMIZE], "test-study")

    # Test existing study
    assert storage.get_study_id_by_name_if_exists("test-study")


def test_get_study_id_by_name_if_exists(setup_storage):
    storage = setup_storage

    # Test non-existent study
    assert storage.get_study_id_by_name_if_exists("non-existent") is None

    # Create a study
    study_id = storage.create_new_study([StudyDirection.MINIMIZE], "test-study")

    # Test existing study
    result = storage.get_study_id_by_name_if_exists("test-study")
    assert result == study_id
