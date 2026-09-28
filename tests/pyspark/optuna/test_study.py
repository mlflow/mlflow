import logging
import os

import numpy as np
import pyspark
import pytest
from optuna.exceptions import TrialPruned
from optuna.pruners import NopPruner, ThresholdPruner
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from packaging.version import Version

import mlflow
from mlflow.exceptions import ExecutionException
from mlflow.pyspark.optuna.study import MlflowSparkStudy

from tests.optuna.test_storage import setup_storage  # noqa: F401
from tests.pyfunc.test_spark import get_spark_session

_logger = logging.getLogger(__name__)


def _first_trial_infeasible(trial):
    return (1.0 if trial.number == 0 else -1.0,)


def _get_spark_session_with_retry(max_tries=3):
    conf = pyspark.SparkConf()
    for attempt in range(max_tries):
        try:
            return get_spark_session(conf)
        except Exception as e:
            if attempt >= max_tries - 1:
                raise
            _logger.exception(
                f"Attempt {attempt} to create a SparkSession failed ({e!r}), retrying..."
            )


# Specify `autouse=True` to ensure that a context is created
# before any tests are executed. This ensures that the Hadoop filesystem
# does not create its own SparkContext without the MLeap libraries required by
# other tests.
@pytest.fixture(scope="module", autouse=True)
def spark():
    if Version(pyspark.__version__) < Version("3.1"):
        spark_home = (
            os.environ.get("SPARK_HOME")
            if "SPARK_HOME" in os.environ
            else os.path.dirname(pyspark.__file__)
        )
        conf_dir = os.path.join(spark_home, "conf")
        os.makedirs(conf_dir, exist_ok=True)
        with open(os.path.join(conf_dir, "spark-defaults.conf"), "w") as f:
            conf = """
spark.driver.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"
spark.executor.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"
"""
            f.write(conf)

    with _get_spark_session_with_retry() as spark:
        yield spark


def test_study_optimize_run(setup_storage):
    storage = setup_storage
    study_name = "test-study"
    sampler = TPESampler(seed=10)
    mlflow_study = MlflowSparkStudy(
        study_name, storage, sampler=sampler, mlflow_tracking_uri=mlflow.get_tracking_uri()
    )

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    mlflow_study.optimize(objective, n_trials=8, n_jobs=4)
    assert sorted(mlflow_study.best_params.keys()) == ["x"]
    assert len(mlflow_study.trials) == 8
    np.testing.assert_allclose(mlflow_study.best_params["x"], 5.426412865334919, rtol=1e-6)


def test_study_with_failed_objective(setup_storage):
    storage = setup_storage
    study_name = "test-study"
    sampler = TPESampler(seed=10)
    mlflow_study = MlflowSparkStudy(
        study_name, storage, sampler=sampler, mlflow_tracking_uri=mlflow.get_tracking_uri()
    )

    def fail_objective(_):
        raise ValueError()

    with pytest.raises(
        ExecutionException,
        match="Optimization run for Optuna MlflowSparkStudy failed",
    ):
        mlflow_study.optimize(fail_objective, n_trials=4)


def test_auto_resume_existing_study(setup_storage):
    storage = setup_storage
    study_name = "resume-test-study"
    sampler = TPESampler(seed=42)

    # Create first study and run some trials
    study1 = MlflowSparkStudy(study_name, storage, sampler=sampler)
    assert not study1.is_resumed_study

    def objective(trial):
        return trial.suggest_float("x", 0, 10) ** 2

    study1.optimize(objective, n_trials=3, n_jobs=1)
    first_trial_count = len(study1.trials)
    first_best_value = study1.best_value

    # Create second study with same name - should resume
    study2 = MlflowSparkStudy(study_name, storage, sampler=sampler)
    assert study2.is_resumed_study
    assert len(study2.trials) == first_trial_count
    assert study2.best_value == first_best_value

    # Continue optimization
    study2.optimize(objective, n_trials=2, n_jobs=1)
    assert len(study2.trials) == first_trial_count + 2

    # Assert that the resumed study generates a better (lower) objective value than the first study
    assert study2.best_value <= first_best_value


def test_new_study_is_not_resumed(setup_storage):
    storage = setup_storage
    study_name = "new-study"

    study = MlflowSparkStudy(study_name, storage)
    assert not study.is_resumed_study
    assert study.completed_trials_count == 0

    info = study.get_resume_info()
    assert not info.is_resumed


def test_resume_info_method(setup_storage):
    storage = setup_storage
    study_name = "info-test-study"
    sampler = TPESampler(seed=123)

    # New study
    study1 = MlflowSparkStudy(study_name, storage, sampler=sampler)
    info = study1.get_resume_info()
    assert not info.is_resumed

    # Run some trials
    def objective(trial):
        return trial.suggest_float("x", 0, 1) ** 2

    study1.optimize(objective, n_trials=2, n_jobs=1)

    # Resume study
    study2 = MlflowSparkStudy(study_name, storage, sampler=sampler)
    info = study2.get_resume_info()
    assert info.is_resumed
    assert info.study_name == study_name
    assert info.existing_trials == 2
    assert info.completed_trials == 2
    assert hasattr(info, "best_value")
    assert hasattr(info, "best_params")
    assert info.best_value is not None
    assert info.best_params is not None


def test_completed_trials_count_property(setup_storage):
    storage = setup_storage
    study_name = "count-test-study"

    study = MlflowSparkStudy(study_name, storage)
    assert study.completed_trials_count == 0

    def objective(trial):
        return trial.suggest_float("x", 0, 1)

    study.optimize(objective, n_trials=3, n_jobs=1)
    assert study.completed_trials_count == 3

    # Resume and check count is preserved
    resumed_study = MlflowSparkStudy(study_name, storage)
    assert resumed_study.completed_trials_count == 3


def test_resume_preserves_best_results(setup_storage):
    storage = setup_storage
    study_name = "best-results-study"
    sampler = TPESampler(seed=456)

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    # First optimization
    study1 = MlflowSparkStudy(study_name, storage, sampler=sampler)
    study1.optimize(objective, n_trials=5, n_jobs=1)

    original_best_value = study1.best_value
    original_best_params = study1.best_params.copy()

    # Resume and verify best results are preserved
    study2 = MlflowSparkStudy(study_name, storage, sampler=sampler)
    assert study2.best_value == original_best_value
    assert study2.best_params == original_best_params

    # Continue optimization and verify it can improve
    study2.optimize(objective, n_trials=5, n_jobs=1)

    # Best value should be the same or better (lower for minimization)
    assert study2.best_value <= original_best_value


def test_resume_only_pruned_study(setup_storage):
    storage = setup_storage
    study_name = "only-pruned-study"

    def pruned_objective(trial):
        trial.report(1.0, step=0)
        if trial.should_prune():
            raise TrialPruned()

    study = MlflowSparkStudy(study_name, storage, pruner=ThresholdPruner(upper=0.5))
    study.optimize(pruned_objective, n_trials=1, n_jobs=1)

    resumed_study = MlflowSparkStudy(study_name, storage, pruner=NopPruner())
    info = resumed_study.get_resume_info()
    assert info.completed_trials == 0
    assert info.best_value is None
    assert info.best_params is None

    resumed_study.optimize(lambda _: 0.0, n_trials=1, n_jobs=1)
    assert resumed_study.completed_trials_count == 1


def test_resume_all_infeasible_study(setup_storage):
    storage = setup_storage
    study_name = "all-infeasible-study"

    def always_infeasible(_):
        return (1.0,)

    sampler = TPESampler(seed=123, constraints_func=always_infeasible)

    study = MlflowSparkStudy(study_name, storage, sampler=sampler)
    study.optimize(lambda _: 1.0, n_trials=1, n_jobs=1)

    resumed_study = MlflowSparkStudy(study_name, storage, sampler=sampler)
    info = resumed_study.get_resume_info()
    assert info.completed_trials == 1
    assert info.best_value is None
    assert info.best_params is None

    resumed_study.optimize(lambda _: 1.0, n_trials=1, n_jobs=1)
    assert resumed_study.completed_trials_count == 2


def test_resume_reports_best_feasible_trial(setup_storage):
    storage = setup_storage
    study_name = "mixed-feasibility-study"
    sampler = TPESampler(seed=123, constraints_func=_first_trial_infeasible)

    study = MlflowSparkStudy(study_name, storage, sampler=sampler)
    study._study.optimize(lambda trial: float(trial.number), n_trials=2)

    resumed_study = MlflowSparkStudy(study_name, storage, sampler=sampler)
    info = resumed_study.get_resume_info()
    assert info.completed_trials == 2
    assert info.best_value == 1.0
    assert info.best_params == {}


def test_pruner_threaded_to_driver_and_executor_studies(setup_storage):
    storage = setup_storage
    study_name = "pruner-test-study"
    mlflow_study = MlflowSparkStudy(study_name, storage, pruner=ThresholdPruner(upper=0.5))
    assert isinstance(mlflow_study._study.pruner, ThresholdPruner)

    # The default MedianPruner never prunes the first trials (n_startup_trials=5),
    # so pruned trials prove the supplied pruner reached the executor-side study.
    def objective(trial):
        trial.report(1.0, step=0)
        if trial.should_prune():
            raise TrialPruned()
        return 1.0

    mlflow_study.optimize(objective, n_trials=2, n_jobs=1)
    assert [t.state for t in mlflow_study.trials] == [TrialState.PRUNED, TrialState.PRUNED]

    resumed_study = MlflowSparkStudy(study_name, storage, pruner=NopPruner())
    assert isinstance(resumed_study._study.pruner, NopPruner)


def test_study_direction_maximize(setup_storage):
    storage = setup_storage
    study_name = "maximize-study"
    sampler = TPESampler(seed=10)
    mlflow_study = MlflowSparkStudy(study_name, storage, sampler=sampler, direction="maximize")
    assert mlflow_study.direction == StudyDirection.MAXIMIZE

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    mlflow_study.optimize(objective, n_trials=4, n_jobs=2)
    values = [t.value for t in mlflow_study.trials]
    assert mlflow_study.best_value == max(values)


def test_study_directions_multi_objective(setup_storage):
    storage = setup_storage
    study_name = "multi-objective-study"
    mlflow_study = MlflowSparkStudy(study_name, storage, directions=["minimize", "maximize"])
    assert mlflow_study.directions == [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x, y

    mlflow_study.optimize(objective, n_trials=3, n_jobs=1)
    assert all(len(t.values) == 2 for t in mlflow_study.trials)
    assert len(mlflow_study.best_trials) >= 1

    resumed_study = MlflowSparkStudy(study_name, storage)
    assert resumed_study.directions == [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
    info = resumed_study.get_resume_info()
    assert info.is_resumed
    assert info.existing_trials == 3
    assert info.completed_trials == 3
    assert info.best_value is None
    assert info.best_params is None

    resumed_study.optimize(objective, n_trials=2, n_jobs=1)
    assert len(resumed_study.trials) == 5

    with pytest.raises(ValueError, match="fixed at creation time"):
        MlflowSparkStudy(study_name, storage, directions=["maximize", "minimize"])


def test_direction_inherited_on_resume_and_conflict_raises(setup_storage):
    storage = setup_storage
    study_name = "direction-resume-study"
    MlflowSparkStudy(study_name, storage, direction="maximize")

    resumed_study = MlflowSparkStudy(study_name, storage)
    assert resumed_study.direction == StudyDirection.MAXIMIZE

    same_direction_study = MlflowSparkStudy(study_name, storage, direction=StudyDirection.MAXIMIZE)
    assert same_direction_study.direction == StudyDirection.MAXIMIZE

    with pytest.raises(ValueError, match="fixed at creation time"):
        MlflowSparkStudy(study_name, storage, direction="minimize")

    with pytest.raises(ValueError, match="Invalid value"):
        MlflowSparkStudy(study_name, storage, direction=123)


def test_direction_and_directions_mutually_exclusive(setup_storage):
    storage = setup_storage
    with pytest.raises(ValueError, match="Specify only one of"):
        MlflowSparkStudy("exclusive-study", storage, direction="minimize", directions=["minimize"])


def test_directions_string_raises(setup_storage):
    storage = setup_storage
    with pytest.raises(ValueError, match="must be a sequence"):
        MlflowSparkStudy("directions-string-study", storage, directions="maximize")


def test_empty_directions_raises(setup_storage):
    storage = setup_storage
    message = "The number of objectives must be greater than 0"

    with pytest.raises(ValueError, match=message):
        MlflowSparkStudy("empty-directions-study", storage, directions=[])

    MlflowSparkStudy("existing-study", storage)
    with pytest.raises(ValueError, match=message):
        MlflowSparkStudy("existing-study", storage, directions=[])
