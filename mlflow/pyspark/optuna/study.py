import datetime
import logging
import sys
from collections.abc import Callable, Iterable
from typing import Optional

try:
    import optuna
    from optuna import pruners, samplers, storages
    from optuna.study import Study
    from optuna.study._optimize import _optimize_sequential
    from optuna.trial import FrozenTrial
except ImportError:
    sys.exit()

from pyspark.sql import SparkSession

import mlflow
from mlflow.optuna.storage import MlflowStorage

logger = logging.getLogger("optuna-spark")


def is_spark_connect_mode():
    """Check if the current Spark session is running in client mode."""
    try:
        from pyspark.sql.utils import is_remote
    except ImportError:
        return False
    return is_remote()


class MLFlowSparkStudy(Study):
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with spark
    via MLFlow experiment.

    .. code-block:: python
        :caption: Example

        from mlflow.optuna.storage import MlflowStorage
        from mlflow.pyspark.optuna.study import MLFlowSparkStudy

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        experiment_id = "507151065975140"
        study_name = "spark_mlflow_storage"

        storage = MLFlowStorage(experiment_id=experiment_id)
        mlflow_study = MLFlowSparkStudy(
            study_name, storage, mlflow_tracking_uri=mlflow.get_tracking_uri())
        mlflow_study.optimize(objective, n_trials=4)
    """

    def __init__(
        self,
        study_name: str,
        storage: MlflowStorage,
        sampler: Optional["samplers.BaseSampler"] = None,
        pruner: Optional[pruners.BasePruner] = None,
        mlflow_tracking_uri: Optional[str] = "databricks",
    ):
        self.study_name = study_name
        self._storage = storages.get_storage(storage)
        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self.spark = SparkSession.active()

        # check whether the SparkConnect mode
        self._is_spark_connect_mode = is_spark_connect_mode()
        self._mlflow_tracking_env = mlflow_tracking_uri

        mlflow.set_tracking_uri(self._mlflow_tracking_env)
        self._study = optuna.create_study(
            study_name=self.study_name, sampler=self.sampler, storage=self._storage
        )
        self._study_id = storage.get_study_id_from_name(self.study_name)
        self._directions = self._storage.get_study_directions(self._study_id)

        if not isinstance(self._storage, MlflowStorage):
            raise ValueError(
                f"MLFlowSparkStudy only works with `MlflowStorage`. But get {type(self._storage)}."
            )

    def optimize(
        self,
        func: "optuna.study.study.ObjectiveFuncType",
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Iterable[type[Exception]] = (),
        callbacks: Optional[Iterable[Callable[[Study, FrozenTrial], None]]] = None,
    ) -> None:
        experiment_id = self._storage._experiment_id
        study_name = self.study_name
        mlflow_tracking_env = self._mlflow_tracking_env
        sampler = self.sampler

        def run_task_on_executor_pd(iterator):
            import traceback

            import optuna
            import pandas as pd

            import mlflow
            from mlflow.optuna.storage import MlflowStorage

            mlflow.set_tracking_uri(mlflow_tracking_env)

            storage = MlflowStorage(experiment_id=experiment_id)
            study = optuna.load_study(study_name=study_name, sampler=sampler, storage=storage)
            num_trials = sum(map(len, iterator))

            try:
                error_message = []
                _optimize_sequential(
                    study,
                    func,
                    num_trials,
                    timeout,
                    catch,
                    callbacks,
                    False,
                    False,
                    datetime.datetime.now(),
                    None,
                )
                error_message.append(None)
            except BaseException:
                _traceback_string = traceback.format_exc()
                error_message.append(_traceback_string)
                mlflow.set_tag("error_message", error_message)
                mlflow.set_tag("LOG_STATUS", "FAILED")
                raise
            finally:
                df = pd.DataFrame(
                    {
                        "error": error_message,
                    }
                )
                yield df

        num_tasks = n_trials
        if n_jobs == -1:
            n_jobs = num_tasks
        input_df = self.spark.range(start=0, end=num_tasks, step=1, numPartitions=n_jobs)
        trial_tag = f"optuna_trial_{study_name}_{experiment_id}"
        if self._is_spark_connect_mode:
            self.spark.addTag(trial_tag)

        try:
            input_df.mapInPandas(
                func=run_task_on_executor_pd,
                schema="error string",
            ).collect()
        except KeyboardInterrupt as e:
            if self._is_spark_connect_mode:
                self.spark.interruptTag(trial_tag)
            logger.debug("MLFlowSparkStudy.optimize terminated by user.")
            raise e
