import datetime
import sys
from collections.abc import Callable, Iterable
from typing import Optional

try:
    import optuna
    from optuna import storages
    from optuna.study import Study
    from optuna.study._optimize import _optimize_sequential
    from optuna.trial import FrozenTrial
except ImportError:
    sys.exit()

from pyspark.sql import SparkSession

from mlflow.pyspark.optuna.storage import MlflowStorage


class MLFlowSparkStudy:
    """A wrapper of :class:`~optuna.study.Study` to incorporate Optuna with spark
    via MLFlow experiment.
    """

    def __init__(self, study_name: str, storage: MlflowStorage):
        self._study_name = study_name
        self._storage = storages.get_storage(storage)

        self.spark = SparkSession.active()

        if not isinstance(self._storage, MlflowStorage):
            raise ValueError(f"MLFlowSparkStudy only works with `MlflowStorage`. But get {type(self._storage)}.")

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
        study_name = self._study_name

        def run_task_on_executor_pd(iterator):
            import traceback

            import optuna
            import pandas as pd

            import mlflow
            from mlflow.pyspark.optuna.storage import MLFlowStorage

            mlflow.set_tracking_uri("databricks")

            try:
                storage = MLFlowStorage(experiment_id=experiment_id)
                study = optuna.load_study(study_name=study_name, storage=storage)
                _optimize_sequential(
                    study,
                    func,
                    1,
                    timeout,
                    catch,
                    callbacks,
                    False,
                    False,
                    datetime.datetime.now(),
                    None,
                )
                error_message = None
            except BaseException as e:
                _traceback_string = traceback.format_exc()
                e._tb_str = _traceback_string
                error_message = _traceback_string
            finally:
                df = pd.DataFrame(
                    {
                        "error": [error_message],
                    }
                )
                yield df

        num_tasks = n_trials
        if n_jobs == -1:
            n_jobs = num_tasks
        input_df = self.spark.range(start=0, end=num_tasks, step=1, numPartitions=n_jobs)

        return input_df.mapInPandas(
            func=run_task_on_executor_pd,
            schema="error string",
        ).collect()

