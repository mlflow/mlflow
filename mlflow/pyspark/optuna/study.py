import datetime
import logging
import tempfile
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna import exceptions, pruners, samplers, storages
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import ExecutionException
from mlflow.optuna.storage import MlflowStorage

_logger = logging.getLogger(__name__)


@dataclass
class ResumeInfo:
    is_resumed: bool
    study_name: str | None = None
    existing_trials: int | None = None
    completed_trials: int | None = None
    best_value: float | None = None
    best_params: dict[str, Any] | None = None


def is_spark_connect_mode() -> bool:
    """Check if the current Spark session is running in client mode."""
    try:
        from pyspark.sql.utils import is_remote
    except ImportError:
        return False
    return is_remote()


def _optimize_sequential(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    mlflow_client: MlflowClient,
    n_trials: int = 1,
    timeout: float | None = None,
    catch: Iterable[type[Exception]] = (),
    callbacks: Iterable[Callable[[Study, FrozenTrial], None]] | None = None,
) -> None:
    """
    Run optimization sequentially. It is modified from _optimize_sequential in optuna
    (https://github.com/optuna/optuna/blob/e1e30e7150047e5f582b8fef1eeb65386cb1c4c1/optuna/study/_optimize.py#L121)
    Convert the nested call to one function and log the error messages to mlflow.
    """
    i_trial = 0
    time_start = datetime.datetime.now()

    while True:
        if study._stop_flag:
            break
        if i_trial >= n_trials:
            break
        i_trial += 1

        if timeout is not None:
            elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
            if elapsed_seconds >= timeout:
                break

        state = None
        value_or_values = None
        func_err = None
        func_err_fail_exc_info = None
        trial = study.ask()

        try:
            value_or_values = func(trial)
        except exceptions.TrialPruned as e:
            state = TrialState.PRUNED
            func_err = e
        except (Exception, KeyboardInterrupt) as e:
            state = TrialState.FAIL
            func_err = e
            func_err_fail_exc_info = traceback.format_exc()
        try:
            frozen_trial, warning_message = optuna.study._tell._tell_with_warning(
                study=study,
                trial=trial,
                value_or_values=value_or_values,
                state=state,
                suppress_warning=True,
            )
        except Exception:
            frozen_trial = study._storage.get_trial(trial._trial_id)
            warning_message = None

        if frozen_trial.state == TrialState.COMPLETE:
            _logger.info(f"Trial {trial.number} finished with parameters: {trial.params}.")
        elif frozen_trial.state == TrialState.PRUNED:
            _logger.info("Trial {} pruned. {}".format(frozen_trial._trial_id, str(func_err)))
            mlflow_client.set_terminated(frozen_trial._trial_id, status="KILLED")
        elif frozen_trial.state == TrialState.FAIL:
            error_message = None
            if func_err is not None:
                error_message = func_err_fail_exc_info
            elif warning_message is not None:
                error_message = warning_message
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir, "error_message.txt")
                path.write_text(error_message)
                # Log the file as an artifact in the active MLflow run
                mlflow_client.log_artifact(frozen_trial._trial_id, path)
                mlflow_client.set_terminated(frozen_trial._trial_id, status="FAILED")

        if (
            frozen_trial.state == TrialState.FAIL
            and func_err is not None
            and not isinstance(func_err, catch)
        ):
            raise func_err

        if callbacks is not None:
            for callback in callbacks:
                callback(study, frozen_trial)


class MlflowSparkStudy(Study):
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with spark
    via MLflow experiment.

    This class automatically resumes existing studies with the same name,
    allowing for interrupted optimization to continue from where it left off.

    .. code-block:: python
        :caption: Basic Usage

        from mlflow.optuna.storage import MlflowStorage
        from mlflow.pyspark.optuna.study import MlflowSparkStudy


        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2


        experiment_id = "507151065975140"
        study_name = "spark_mlflow_storage"

        storage = MlflowStorage(experiment_id=experiment_id)
        mlflow_study = MlflowSparkStudy(study_name, storage)
        mlflow_study.optimize(objective, n_trials=4)

        # Later, create another instance with same name to resume
        resumed_study = MlflowSparkStudy(study_name, storage)
        print(f"Resumed with {len(resumed_study.trials)} existing trials")
        resumed_study.optimize(objective, n_trials=4)  # Continue optimization
    """

    def __init__(
        self,
        study_name: str,
        storage: MlflowStorage,
        sampler: samplers.BaseSampler | None = None,
        pruner: pruners.BasePruner | None = None,
        mlflow_tracking_uri: str | None = None,
    ):
        self.study_name = study_name
        self._storage = storages.get_storage(storage)
        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self.spark = SparkSession.active()

        # check whether the SparkConnect mode
        self._is_spark_connect_mode = is_spark_connect_mode()
        self._mlflow_tracking_env = mlflow_tracking_uri or mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self._mlflow_tracking_env)
        self.mlflow_client = MlflowClient()

        if not isinstance(self._storage, MlflowStorage):
            raise ValueError(
                f"MlflowSparkStudy only works with `MlflowStorage`. But get {type(self._storage)}."
            )

        # Check if study exists and auto-resume if it does
        if self._storage.get_study_id_by_name_if_exists(self.study_name):
            # Load existing study
            self._study = optuna.load_study(
                study_name=self.study_name, sampler=self.sampler, storage=self._storage
            )
            self._study_id = self._storage.get_study_id_from_name(self.study_name)
            self._is_resumed = True
            _logger.info(
                f"Resuming existing study '{self.study_name}' with {len(self._study.trials)} trials"
            )
        else:
            # Create new study
            self._study = optuna.create_study(
                study_name=self.study_name, sampler=self.sampler, storage=self._storage
            )
            self._study_id = self._storage.get_study_id_from_name(self.study_name)
            self._is_resumed = False
            _logger.info(f"Created new study '{self.study_name}'")

        self._directions = self._storage.get_study_directions(self._study_id)

    @property
    def is_resumed_study(self) -> bool:
        """Check if this study was resumed from existing data.

        Returns:
            True if the study was resumed from existing data, False if it's a new study
        """
        return self._is_resumed

    @property
    def completed_trials_count(self) -> int:
        """Number of completed trials in the study.

        Returns:
            Count of trials that have completed successfully
        """
        return len([t for t in self._study.trials if t.state == TrialState.COMPLETE])

    def get_resume_info(self) -> ResumeInfo | None:
        """Get information about the resumed study.

        Returns:
            ResumeInfo dataclass containing resume information including trial
            counts and best results
        """
        if not self._is_resumed:
            return ResumeInfo(is_resumed=False)

        return ResumeInfo(
            is_resumed=True,
            study_name=self.study_name,
            existing_trials=len(self._study.trials),
            completed_trials=self.completed_trials_count,
            best_value=self._study.best_value if self._study.trials else None,
            best_params=self._study.best_params if self._study.trials else None,
        )

    def optimize(
        self,
        func: "optuna.study.study.ObjectiveFuncType",
        n_trials: int | None = None,
        timeout: float | None = None,
        n_jobs: int = -1,
        catch: Iterable[type[Exception]] = (),
        callbacks: Iterable[Callable[[Study, FrozenTrial], None]] | None = None,
    ) -> None:
        # Add logging for resume information
        if self._is_resumed and self._study.trials:
            _logger.info(f"""
            Continuing optimization with {len(self._study.trials)} existing trials.
            Current best value: {self._study.best_value}
            """)
        elif self._is_resumed:
            _logger.info("Resuming study with no previous trials")
        else:
            _logger.info("Starting optimization for new study")

        experiment_id = self._storage._experiment_id
        study_name = self.study_name
        mlflow_tracking_env = self._mlflow_tracking_env
        sampler = self.sampler

        def run_task_on_executor_pd(iterator):
            mlflow.set_tracking_uri(mlflow_tracking_env)
            mlflow_client = MlflowClient()

            storage = MlflowStorage(experiment_id=experiment_id)
            study = optuna.load_study(study_name=study_name, sampler=sampler, storage=storage)
            num_trials = sum(map(len, iterator))

            error_message = None
            try:
                _optimize_sequential(
                    study=study,
                    func=func,
                    mlflow_client=mlflow_client,
                    n_trials=num_trials,
                    timeout=timeout,
                    catch=catch,
                    callbacks=callbacks,
                )
            except BaseException:
                error_message = traceback.format_exc()
            yield pd.DataFrame({"error": [error_message]})

        num_tasks = n_trials
        if n_jobs == -1:
            n_jobs = num_tasks
        input_df = self.spark.range(start=0, end=num_tasks, step=1, numPartitions=n_jobs)
        trial_tag = f"optuna_trial_{study_name}_{experiment_id}"
        if self._is_spark_connect_mode:
            self.spark.addTag(trial_tag)
        else:
            job_group_id = self.spark.sparkContext.getLocalProperty("spark.jobGroup.id")
            if job_group_id is None:
                job_group_id = trial_tag
                job_group_description = f"optuna_trial_{study_name}"
                self.spark.sparkContext.setJobGroup(
                    job_group_id, job_group_description, interruptOnCancel=True
                )
        try:
            result_df = input_df.mapInPandas(
                func=run_task_on_executor_pd,
                schema="error string",
            )
        except KeyboardInterrupt:
            if self._is_spark_connect_mode:
                self.spark.interruptTag(trial_tag)
            else:
                self.spark.sparkContext.cancelJobGroup(trial_tag)
            _logger.debug("MlflowSparkStudy optimize terminated by user.")
            self.mlflow_client.set_terminated(self._study_id, "KILLED")
            raise
        if "error" in result_df.columns:
            failed_runs = result_df.filter(col("error").isNotNull())
            error_rows = failed_runs.select("error").collect()
            if len(error_rows) > 0:
                first_non_null_value = error_rows[0][0]
                self.mlflow_client.set_terminated(self._study_id, "KILLED")
                raise ExecutionException(
                    f"Optimization run for Optuna MlflowSparkStudy failed. "
                    f"See full error details in the failed MLflow runs. "
                    f"Number of failed runs: {len(error_rows)}. "
                    f"First trial failure message: {first_non_null_value}"
                )
        self.mlflow_client.set_terminated(self._study_id)
