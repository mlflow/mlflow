from datetime import datetime
import logging
import os
import random
import tempfile
import unittest
from time import sleep
import numpy as np
from typing import Any, Tuple, List, Dict

import mlflow

from mlflow.pyspark.optuna.storage import MLFlowStorage

from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.storages import BaseStorage
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

ALL_STATES = list(TrialState)

EXAMPLE_ATTRS: dict[str, Any] = {
    "dataset": "MNIST",
    "none": None,
    "json_serializable": {
        "baseline_score": 0.001,
        "tags": ["image", "classification"]
    },
}


def _setup_studies(
        storage: BaseStorage,
        n_study: int,
        n_trial: int,
        seed: int,
        direction: StudyDirection = None,
) -> Tuple[Dict[int, FrozenStudy], Dict[int, Dict[int, FrozenTrial]]]:
    generator = random.Random(seed)
    study_id_to_frozen_study: Dict[int, FrozenStudy] = {}
    study_id_to_trials: Dict[int, dict[int, FrozenTrial]] = {}
    for i in range(n_study):
        study_name = "test-study-name-{}".format(i)
        if direction is None:
            direction = generator.choice([StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
        study_id = storage.create_new_study(directions=(direction, ), study_name=study_name)
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
        "attrD": {
            "baseline_score": 0.001,
            "tags": ["image", "classification"]
        },
    }
    state = generator.choice(ALL_STATES)
    params = {}
    distributions = {}
    user_attrs = {}
    system_attrs: Dict[str, Any] = {}
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


class OptunaMlfowTestSuite(unittest.TestCase):
    @classmethod
    def set_up_class_for_mlflow(cls, temp_dir="/tmp"):
        cls.tempdir = tempfile.mkdtemp(prefix="optuna_tests_", dir=temp_dir)
        cls.mlflow_uri = "file:" + os.path.join(cls.tempdir, "mlflow")
        mlflow.set_tracking_uri(cls.mlflow_uri)
        logging.info("{test} logging to MLflow URI: {uri}".format(
            test=cls.__name__, uri=cls.mlflow_uri))

    def setUp(self):
        self.set_up_class_for_mlflow()
        self.experiment_id = mlflow.create_experiment(name="optuna_mlflow_test")
        self.storage = MLFlowStorage(experiment_id=self.experiment_id)

    def tearDown(self):
        mlflow.delete_experiment(self.experiment_id)

    def test_create_new_study(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        frozen_studies = self.storage.get_all_studies()
        assert len(frozen_studies) == 1
        assert frozen_studies[0]._study_id == study_id
        assert frozen_studies[0].study_name.startswith(DEFAULT_STUDY_NAME_PREFIX)

        study_id2 = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        # Study id must be unique.
        assert study_id != study_id2
        frozen_studies = self.storage.get_all_studies()
        assert len(frozen_studies) == 2
        assert {s._study_id for s in frozen_studies} == {study_id, study_id2}
        assert all(s.study_name.startswith(DEFAULT_STUDY_NAME_PREFIX) for s in frozen_studies)

    def test_delete_study(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        self.storage.create_new_trial(study_id)
        trials = self.storage.get_all_trials(study_id)
        assert len(trials) == 1

        with self.assertRaises(mlflow.exceptions.MlflowException):
            # Deletion of non-existent study.
            self.storage.delete_study(study_id + "1")

        self.storage.delete_study(study_id)

    def test_get_study_id_from_name_and_get_study_name_from_id(self):

        study_name = "test_optuna_mlflow_study"
        study_id = self.storage.create_new_study(
            directions=[StudyDirection.MINIMIZE], study_name=study_name)

        # Test existing study.
        assert self.storage.get_study_name_from_id(study_id) == study_name
        assert self.storage.get_study_id_from_name(study_name) == study_id

        # Test not existing study.
        with self.assertRaises(Exception):
            self.storage.get_study_id_from_name("dummy-name")

        with self.assertRaises(mlflow.exceptions.MlflowException):
            self.storage.get_study_name_from_id(study_id + "1")

    def test_set_and_get_study_directions(self):
        for target in [
            (StudyDirection.MINIMIZE, ),
            (StudyDirection.MAXIMIZE, ),
            (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE),
            (StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE),
            [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE],
            [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE],
        ]:
            study_id = self.storage.create_new_study(directions=target)

            def check_get() -> None:
                got_directions = self.storage.get_study_directions(study_id)

                assert got_directions == list(
                    target), "Direction of a study should be a tuple of `StudyDirection` objects."

            # Test setting value.
            check_get()

            # Test non-existent study.
            non_existent_study_id = study_id + "1"

            with self.assertRaises(mlflow.exceptions.MlflowException):
                self.storage.get_study_directions(non_existent_study_id)

    def test_set_and_get_study_user_attrs(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        def check_set_and_get(key: str, value: Any) -> None:
            self.storage.set_study_user_attr(study_id, key, value)
            assert self.storage.get_study_user_attrs(study_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(key, value)
        assert self.storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get("dataset", "ImageNet")

        # Non-existent study id.
        non_existent_study_id = study_id + "1"
        with self.assertRaises(mlflow.exceptions.MlflowException):
            self.storage.get_study_user_attrs(non_existent_study_id)

        # Non-existent study id.
        with self.assertRaises(mlflow.exceptions.MlflowException):
            self.storage.set_study_user_attr(non_existent_study_id, "key", "value")

    def test_set_and_get_study_system_attrs(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        def check_set_and_get(key: str, value: Any) -> None:
            self.storage.set_study_system_attr(study_id, key, value)
            assert self.storage.get_study_system_attrs(study_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(key, value)
        assert self.storage.get_study_system_attrs(study_id) == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get("dataset", "ImageNet")

    def test_create_new_trial(self):
        def _check_trials(
                trials: List[FrozenTrial],
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
            assert all(t.datetime_start < time_before_creation for t in trials
                       if t._trial_id != trial_id and t.datetime_start is not None)
            assert all(time_before_creation < t.datetime_start < time_after_creation for t in trials
                       if t._trial_id == trial_id and t.datetime_start is not None)
            assert all(t.value is None for t in trials)

        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            time_before_creation = datetime.now()
            sleep(0.001)  # Sleep 1ms to avoid faulty assertion on Windows OS.
            trial_id = self.storage.create_new_trial(study_id)
            sleep(0.001)
            time_after_creation = datetime.now()

            trials = self.storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id, time_before_creation, time_after_creation)

    def test_create_new_trial_with_template_trial(self):
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
            intermediate_values={
                1: 10,
                2: 100,
                3: 1000
            },
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

        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            trial_id = self.storage.create_new_trial(study_id, template_trial=template_trial)
            trials = self.storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id)

    def test_get_trial_number_from_id(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id = self.storage.create_new_trial(study_id)
        assert self.storage.get_trial_number_from_id(trial_id) == 0

        trial_id = self.storage.create_new_trial(study_id)
        assert self.storage.get_trial_number_from_id(trial_id) == 1

    def test_set_trial_state_values_for_state(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_ids = [self.storage.create_new_trial(study_id) for _ in ALL_STATES]

        for trial_id, state in zip(trial_ids, ALL_STATES):
            if state == TrialState.WAITING:
                continue
            assert self.storage.get_trial(trial_id).state == TrialState.RUNNING
            datetime_start_prev = self.storage.get_trial(trial_id).datetime_start
            self.storage.set_trial_state_values(
                trial_id, state=state, values=(0.0, ) if state.is_finished() else None)
            assert self.storage.get_trial(trial_id).state == state
            # Repeated state changes to RUNNING should not trigger further datetime_start changes.
            if state == TrialState.RUNNING:
                assert self.storage.get_trial(trial_id).datetime_start == datetime_start_prev
            if state.is_finished():
                assert self.storage.get_trial(trial_id).datetime_complete is not None
            else:
                assert self.storage.get_trial(trial_id).datetime_complete is None

    def test_get_trial_param_and_get_trial_params(self):
        _, study_to_trials = _setup_studies(self.storage, n_study=2, n_trial=5, seed=1)

        for _, trial_id_to_trial in study_to_trials.items():
            for trial_id, expected_trial in trial_id_to_trial.items():
                assert self.storage.get_trial_params(trial_id) == expected_trial.params
                for key in expected_trial.params.keys():
                    assert self.storage.get_trial_param(
                        trial_id, key) == expected_trial.distributions[key].to_internal_repr(
                        expected_trial.params[key])

    def test_set_trial_param(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id_1 = self.storage.create_new_trial(study_id)
        trial_id_2 = self.storage.create_new_trial(study_id)

        # Setup distributions.
        distribution_x = FloatDistribution(low=1.0, high=2.0)
        distribution_y_1 = CategoricalDistribution(choices=("Shibuya", "Ebisu", "Meguro"))
        distribution_z = FloatDistribution(low=1.0, high=100.0, log=True)

        # Set new params.
        self.storage.set_trial_param(trial_id_1, "x", 0.5, distribution_x)
        self.storage.set_trial_param(trial_id_1, "y", 2, distribution_y_1)
        assert self.storage.get_trial_param(trial_id_1, "x") == 0.5
        assert self.storage.get_trial_param(trial_id_1, "y") == 2
        # Check set_param breaks neither get_trial nor get_trial_params.
        assert self.storage.get_trial(trial_id_1).params == {"x": 0.5, "y": "Meguro"}
        assert self.storage.get_trial_params(trial_id_1) == {"x": 0.5, "y": "Meguro"}

        # Set params to another trial.
        self.storage.set_trial_param(trial_id_2, "x", 0.3, distribution_x)
        self.storage.set_trial_param(trial_id_2, "z", 0.1, distribution_z)
        assert self.storage.get_trial_param(trial_id_2, "x") == 0.3
        assert self.storage.get_trial_param(trial_id_2, "z") == 0.1
        assert self.storage.get_trial(trial_id_2).params == {"x": 0.3, "z": 0.1}
        assert self.storage.get_trial_params(trial_id_2) == {"x": 0.3, "z": 0.1}

        self.storage.set_trial_state_values(trial_id_2, state=TrialState.COMPLETE)
        # Cannot assign params to finished trial.
        with self.assertRaises(RuntimeError):
            self.storage.set_trial_param(trial_id_2, "y", 2, distribution_y_1)
        # Check the previous call does not change the params.
        with self.assertRaises(KeyError):
            self.storage.get_trial_param(trial_id_2, "y")
        # State should be checked prior to distribution compatibility.
        with self.assertRaises(RuntimeError):
            self.storage.set_trial_param(trial_id_2, "y", 0.4, distribution_z)

    def test_set_trial_state_values_for_values(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id_1 = self.storage.create_new_trial(study_id)
        trial_id_2 = self.storage.create_new_trial(study_id)
        trial_id_3 = self.storage.create_new_trial(
            self.storage.create_new_study(directions=[StudyDirection.MINIMIZE]))
        trial_id_4 = self.storage.create_new_trial(study_id)
        trial_id_5 = self.storage.create_new_trial(study_id)

        # Test setting new value.
        self.storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE, values=(0.5, ))
        self.storage.set_trial_state_values(
            trial_id_3, state=TrialState.COMPLETE, values=(float("inf"), ))
        self.storage.set_trial_state_values(
            trial_id_4, state=TrialState.WAITING, values=(0.1, 0.2, 0.3))
        self.storage.set_trial_state_values(
            trial_id_5, state=TrialState.WAITING, values=[0.1, 0.2, 0.3])

        assert self.storage.get_trial(trial_id_1).value == 0.5
        assert self.storage.get_trial(trial_id_2).value is None
        assert self.storage.get_trial(trial_id_3).value == float("inf")
        assert self.storage.get_trial(trial_id_4).values == [0.1, 0.2, 0.3]
        assert self.storage.get_trial(trial_id_5).values == [0.1, 0.2, 0.3]

    def test_set_trial_intermediate_value(self):
        study_id = self.storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id_1 = self.storage.create_new_trial(study_id)
        trial_id_2 = self.storage.create_new_trial(study_id)
        trial_id_3 = self.storage.create_new_trial(
            self.storage.create_new_study(directions=[StudyDirection.MINIMIZE]))
        trial_id_4 = self.storage.create_new_trial(study_id)

        # Test setting new values.
        self.storage.set_trial_intermediate_value(trial_id_1, 0, 0.3)
        self.storage.set_trial_intermediate_value(trial_id_1, 2, 0.4)
        self.storage.set_trial_intermediate_value(trial_id_3, 0, 0.1)
        self.storage.set_trial_intermediate_value(trial_id_3, 1, 0.4)
        self.storage.set_trial_intermediate_value(trial_id_3, 2, 0.5)
        self.storage.set_trial_intermediate_value(trial_id_3, 3, float("inf"))
        self.storage.set_trial_intermediate_value(trial_id_4, 0, float("nan"))

        assert self.storage.get_trial(trial_id_1).intermediate_values == {0: 0.3, 2: 0.4}
        assert self.storage.get_trial(trial_id_2).intermediate_values == {}
        assert self.storage.get_trial(trial_id_3).intermediate_values == {
            0: 0.1,
            1: 0.4,
            2: 0.5,
            3: float("inf"),
        }
        assert np.isnan(self.storage.get_trial(trial_id_4).intermediate_values[0])

        # Test setting existing step.
        self.storage.set_trial_intermediate_value(trial_id_1, 0, 0.2)
        assert self.storage.get_trial(trial_id_1).intermediate_values == {0: 0.2, 2: 0.4}

    def test_get_trial_user_attrs(self):
        _, study_to_trials = _setup_studies(self.storage, n_study=2, n_trial=5, seed=10)
        assert all(
            self.storage.get_trial_user_attrs(trial_id) == trial.user_attrs
            for trials in study_to_trials.values() for trial_id, trial in trials.items())

    def test_set_trial_user_attr(self):
        trial_id_1 = self.storage.create_new_trial(
            self.storage.create_new_study(directions=[StudyDirection.MINIMIZE]))

        def check_set_and_get(trial_id: int, key: str, value: Any) -> None:
            self.storage.set_trial_user_attr(trial_id, key, value)
            assert self.storage.get_trial(trial_id).user_attrs[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(trial_id_1, key, value)
        assert self.storage.get_trial(trial_id_1).user_attrs == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get(trial_id_1, "dataset", "ImageNet")

    def test_set_trial_system_attr(self):
        trial_id_1 = self.storage.create_new_trial(
            self.storage.create_new_study(directions=[StudyDirection.MINIMIZE]))

        def check_set_and_get(trial_id: int, key: str, value: Any) -> None:
            self.storage.set_trial_system_attr(trial_id, key, value)
            assert self.storage.get_trial_system_attrs(trial_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(trial_id_1, key, value)
        system_attrs = self.storage.get_trial(trial_id_1).system_attrs
        assert system_attrs == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get(trial_id_1, "dataset", "ImageNet")

    def test_get_n_trials(self):
        study_id_to_frozen_studies, _ = _setup_studies(self.storage, n_study=2, n_trial=7, seed=50)
        for study_id in study_id_to_frozen_studies:
            assert self.storage.get_n_trials(study_id) == 7


if __name__ == "__main__":
    from databricks.ml.tests.test_mlflow_storage import *  # noqa: F401

    try:
        import xmlrunner

        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
