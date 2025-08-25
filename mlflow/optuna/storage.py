import copy
import datetime
import json
import threading
import time
import uuid
from collections.abc import Container, Sequence
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

try:
    from optuna._typing import JSONSerializable
    from optuna.distributions import (
        BaseDistribution,
        check_distribution_compatibility,
        distribution_to_json,
        json_to_distribution,
    )
    from optuna.storages import BaseStorage
    from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
    from optuna.study import StudyDirection
    from optuna.study._frozen import FrozenStudy
    from optuna.trial import FrozenTrial, TrialState
except ImportError as e:
    raise ImportError("Install optuna to use `mlflow.optuna` module") from e

optuna_mlflow_status_map = {
    TrialState.RUNNING: "RUNNING",
    TrialState.COMPLETE: "FINISHED",
    TrialState.PRUNED: "KILLED",
    TrialState.FAIL: "FAILED",
    TrialState.WAITING: "SCHEDULED",
}

mlflow_optuna_status_map = {
    "RUNNING": TrialState.RUNNING,
    "FINISHED": TrialState.COMPLETE,
    "KILLED": TrialState.PRUNED,
    "FAILED": TrialState.FAIL,
    "SCHEDULED": TrialState.WAITING,
}


class MlflowStorage(BaseStorage):
    """
    MLflow based storage class with batch processing to avoid REST API throttling.
    """

    def __init__(
        self,
        experiment_id: str,
        name: str | None = None,
        batch_flush_interval: float = 1.0,
        batch_size_threshold: int = 100,
    ):
        """
        Initialize MLFlowStorage with batching capabilities.

        Parameters
        ----------
        experiment_id : str
            MLflow experiment ID
        name : Optional[str]
            Optional name for the storage
        batch_flush_interval : float
            Time in seconds between automatic batch flushes (default: 1.0)
        batch_size_threshold : int
            Maximum number of items in batch before triggering a flush (default: 100)
        """
        if not experiment_id:
            raise Exception("No experiment_id provided. MLFlowStorage cannot create experiments.")

        self._experiment_id = experiment_id
        self._mlflow_client = MlflowClient()
        self._name = name

        # Batching configuration
        self._batch_flush_interval = batch_flush_interval
        self._batch_size_threshold = batch_size_threshold

        # Batching queues for metrics, parameters, and tags
        self._batch_queue = {}  # Dictionary of run_id -> {'metrics': [], 'params': [], 'tags': []}
        self._batch_lock = threading.RLock()
        self._last_flush_time = time.time()

        # Flag to indicate if the worker should stop - must be defined BEFORE starting the thread
        self._stop_worker = False

        # Start a background thread for periodic flushing
        self._flush_thread = threading.Thread(
            target=self._periodic_flush_worker,
            daemon=True,
            name=f"mlflow_optuna_batch_flush_worker_{uuid.uuid4().hex[:8]}",
        )
        self._flush_thread.start()

    def __getstate__(self):
        """
        Prepare the object for serialization by removing non-picklable components.
        This is called when the object is being pickled.
        """
        state = self.__dict__.copy()

        # Remove thread-related attributes that can't be pickled
        state.pop("_batch_lock", None)
        state.pop("_flush_thread", None)

        # Store the configuration but not the actual lock/thread
        state["_thread_running"] = hasattr(self, "_flush_thread") and self._flush_thread.is_alive()

        return state

    def __setstate__(self, state):
        """
        Restore the object after deserialization by recreating non-picklable components.
        This is called when the object is being unpickled.
        """
        # First, update the instance with the pickled state
        self.__dict__.update(state)

        # Recreate the lock
        self._batch_lock = threading.RLock()

        # Don't automatically restart the thread on workers - this would create too many threads
        # Instead, we'll use a manual flush approach in distributed contexts
        self._flush_thread = None

        # If we're on a worker node, we should disable automatic background flushing
        # because it could cause issues with multiple threads trying to write to MLflow
        self._stop_worker = True

    def __del__(self):
        """Ensure all queued data is flushed before destroying the object."""
        # Set the stop flag
        if hasattr(self, "_stop_worker"):
            self._stop_worker = True

        # Join the thread if it exists and is alive
        if hasattr(self, "_flush_thread") and self._flush_thread.is_alive():
            try:
                self._flush_thread.join(timeout=5.0)
            except Exception:
                pass  # Ignore errors during cleanup

        # Flush any remaining data
        if hasattr(self, "_batch_queue"):
            try:
                self.flush_all_batches()
            except Exception:
                pass  # Ignore errors during cleanup

    def _periodic_flush_worker(self):
        """Background worker that periodically flushes batched data."""
        while not self._stop_worker:
            try:
                time.sleep(min(0.1, self._batch_flush_interval / 10))  # Sleep in small increments

                # Check if it's time to flush
                current_time = time.time()
                if current_time - self._last_flush_time >= self._batch_flush_interval:
                    self.flush_all_batches()
                    self._last_flush_time = current_time
            except Exception:
                # Catch any exceptions to prevent thread crashes
                time.sleep(1.0)  # Sleep a bit longer if there was an error

    def _queue_batch_operation(
        self,
        run_id: str,
        metrics: list[Metric] | None = None,
        params: list[Param] | None = None,
        tags: list[RunTag] | None = None,
    ):
        """Queue metrics, parameters, or tags for batched processing."""
        with self._batch_lock:
            if run_id not in self._batch_queue:
                self._batch_queue[run_id] = {"metrics": [], "params": [], "tags": []}

            batch = self._batch_queue[run_id]

            if metrics:
                batch["metrics"].extend(metrics)
            if params:
                batch["params"].extend(params)
            if tags:
                batch["tags"].extend(tags)

            # Check if we've reached the batch size threshold for this run
            batch_size = len(batch["metrics"]) + len(batch["params"]) + len(batch["tags"])
            if batch_size >= self._batch_size_threshold:
                self._flush_batch(run_id)

    def _flush_batch(self, run_id: str):
        """Flush the batch for a specific run_id to MLflow."""
        with self._batch_lock:
            if run_id not in self._batch_queue:
                return

            batch = self._batch_queue[run_id]

            # Only make the API call if there's something to flush
            if batch["metrics"] or batch["params"] or batch["tags"]:
                try:
                    self._mlflow_client.log_batch(
                        run_id, metrics=batch["metrics"], params=batch["params"], tags=batch["tags"]
                    )
                except Exception as e:
                    # If the run doesn't exist, propagate the error
                    if "Run with id=" in str(e) and "not found" in str(e):
                        raise
                    # Otherwise, handle or log the error as needed

                # Clear the batch
                batch["metrics"] = []
                batch["params"] = []
                batch["tags"] = []

    def flush_all_batches(self):
        """Flush all pending batches to MLflow."""
        with self._batch_lock:
            run_ids = list(self._batch_queue.keys())

        # Flush each run's batch
        for run_id in run_ids:
            self._flush_batch(run_id)

    def _search_runs_by_name(self, run_name: str):
        filter_string = f"tags.mlflow.runName = '{run_name}'"
        return self._mlflow_client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=filter_string,
            order_by=["attributes.start_time DESC"],
        )

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        """Create a new study as a mlflow run."""
        study_name = study_name or DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4())
        tags = {
            "mlflow.runName": study_name,
            "optuna.study_direction": ",".join(direction.name for direction in directions),
        }
        study_run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=tags)
        return study_run.info.run_id

    def delete_study(self, study_id) -> None:
        """Delete a study."""
        # Ensure any pending changes are saved before deletion
        self._flush_batch(study_id)
        self._mlflow_client.delete_run(study_id)

    def set_study_user_attr(self, study_id, key: str, value: JSONSerializable) -> None:
        """Register a user-defined attribute as mlflow run tags to a study run."""
        # Verify the run exists first to fail fast if it doesn't
        self._mlflow_client.get_run(study_id)

        # Queue the tag if the run exists
        self._queue_batch_operation(study_id, tags=[RunTag(f"user_{key}", json.dumps(value))])

    def set_study_system_attr(self, study_id, key: str, value: JSONSerializable) -> None:
        """Register a optuna-internal attribute as mlflow run tags to a study run."""
        # Verify the run exists first to fail fast if it doesn't
        self._mlflow_client.get_run(study_id)

        # Queue the tag if the run exists
        self._queue_batch_operation(study_id, tags=[RunTag(f"sys_{key}", json.dumps(value))])

    def get_study_id_from_name(self, study_name: str) -> int:
        # Flush all batches to ensure we have the latest data
        self.flush_all_batches()

        runs = self._search_runs_by_name(study_name)
        if len(runs):
            return runs[0].info.run_id
        else:
            raise Exception(f"Study {study_name} not found")

    def get_study_id_by_name_if_exists(self, study_name: str) -> str | None:
        """Get study ID from name if it exists, otherwise return None.

        Args:
            study_name: The name of the study to look for

        Returns:
            Study ID if found, None otherwise
        """
        # Flush all batches to ensure we have the latest data
        self.flush_all_batches()

        if runs := self._search_runs_by_name(study_name):
            return runs[0].info.run_id
        else:
            return None

    def get_study_name_from_id(self, study_id) -> str:
        # Flush the batch for this study to ensure we have the latest data
        self._flush_batch(study_id)

        run = self._mlflow_client.get_run(study_id)
        return run.data.tags["mlflow.runName"]

    def get_study_directions(self, study_id) -> list[StudyDirection]:
        # Flush the batch for this study to ensure we have the latest data
        self._flush_batch(study_id)

        run = self._mlflow_client.get_run(study_id)
        directions_str = run.data.tags["optuna.study_direction"]
        return [StudyDirection[name] for name in directions_str.split(",")]

    def get_study_user_attrs(self, study_id) -> dict[str, Any]:
        # Flush the batch for this study to ensure we have the latest data
        self._flush_batch(study_id)

        run = self._mlflow_client.get_run(study_id)
        user_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("user_"):
                user_attrs[key[5:]] = json.loads(value)
        return user_attrs

    def get_study_system_attrs(self, study_id) -> dict[str, Any]:
        # Flush the batch for this study to ensure we have the latest data
        self._flush_batch(study_id)

        run = self._mlflow_client.get_run(study_id)
        system_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("sys_"):
                system_attrs[key[4:]] = json.loads(value)
        return system_attrs

    def get_all_studies(self) -> list[FrozenStudy]:
        # Flush all batches to ensure we have the latest data
        self.flush_all_batches()

        runs = self._mlflow_client.search_runs(experiment_ids=[self._experiment_id])
        studies = []
        for run in runs:
            study_id = run.info.run_id
            study_name = run.data.tags["mlflow.runName"]
            directions_str = run.data.tags["optuna.study_direction"]
            directions = [StudyDirection[name] for name in directions_str.split(",")]
            studies.append(
                FrozenStudy(
                    study_name=study_name,
                    direction=None,
                    directions=directions,
                    user_attrs=self.get_study_user_attrs(study_id),
                    system_attrs=self.get_study_system_attrs(study_id),
                    study_id=study_id,
                )
            )
        return studies

    def create_new_trial(self, study_id, template_trial: FrozenTrial | None = None) -> int:
        # Ensure study batch is flushed before creating a new trial
        self._flush_batch(study_id)

        if template_trial:
            frozen = copy.deepcopy(template_trial)
        else:
            frozen = FrozenTrial(
                trial_id=-1,  # dummy value.
                number=-1,  # dummy value.
                state=TrialState.RUNNING,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                value=None,
                intermediate_values={},
                datetime_start=datetime.datetime.now(),
                datetime_complete=None,
            )

        distribution_json = {
            k: distribution_to_json(dist) for k, dist in frozen.distributions.items()
        }
        distribution_str = json.dumps(distribution_json)
        tags = {"param_directions": distribution_str}

        trial_run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=tags)
        trial_id = trial_run.info.run_id

        # Add parent run ID tag
        self._queue_batch_operation(trial_id, tags=[RunTag(MLFLOW_PARENT_RUN_ID, study_id)])

        # Log trial_id metric to study
        hash_id = float(hash(trial_id))
        self._queue_batch_operation(
            study_id, metrics=[Metric("trial_id", hash_id, int(time.time() * 1000), 1)]
        )

        # Ensure study batch is flushed to get accurate metric history
        self._flush_batch(study_id)

        trial_ids = self._mlflow_client.get_metric_history(study_id, "trial_id")
        index = next((i for i, obj in enumerate(trial_ids) if obj.value == hash_id), -1)

        self._queue_batch_operation(trial_id, tags=[RunTag("numbers", str(index))])

        # Set trial state
        state = frozen.state
        if state.is_finished():
            self._mlflow_client.set_terminated(trial_id, status=optuna_mlflow_status_map[state])
        else:
            self._mlflow_client.update_run(trial_id, status=optuna_mlflow_status_map[state])

        timestamp = int(time.time() * 1000)
        metrics = []
        params = []
        tags = []

        # Add metrics
        if frozen.values is not None:
            if len(frozen.values) > 1:
                metrics.extend(
                    [
                        Metric(f"value_{idx}", val, timestamp, 1)
                        for idx, val in enumerate(frozen.values)
                    ]
                )
            else:
                metrics.append(Metric("value", frozen.values[0], timestamp, 1))
        elif frozen.value is not None:
            metrics.append(Metric("value", frozen.value, timestamp, 1))

        # Add intermediate values
        metrics.extend(
            [
                Metric("intermediate_value", val, timestamp, int(k))
                for k, val in frozen.intermediate_values.items()
            ]
        )

        # Add params
        params.extend([Param(k, param) for k, param in frozen.params.items()])

        # Add tags
        tags.extend(
            [RunTag(f"user_{key}", json.dumps(value)) for key, value in frozen.user_attrs.items()]
        )
        tags.extend(
            [RunTag(f"sys_{key}", json.dumps(value)) for key, value in frozen.system_attrs.items()]
        )
        tags.extend(
            [
                RunTag(
                    f"param_internal_val_{k}",
                    json.dumps(frozen.distributions[k].to_internal_repr(param)),
                )
                for k, param in frozen.params.items()
            ]
        )

        # Queue all the data to be sent in batches
        self._queue_batch_operation(trial_id, metrics=metrics, params=params, tags=tags)

        return trial_id

    def set_trial_param(
        self,
        trial_id,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        # Flush the batch for this trial to ensure we have the latest data
        self._flush_batch(trial_id)

        trial_run = self._mlflow_client.get_run(trial_id)
        distributions_dict = json.loads(trial_run.data.tags["param_directions"])
        self.check_trial_is_updatable(trial_id, mlflow_optuna_status_map[trial_run.info.status])

        if param_name in trial_run.data.params:
            param_distribution = json_to_distribution(distributions_dict[param_name])
            check_distribution_compatibility(param_distribution, distribution)

        # Queue parameter update
        self._queue_batch_operation(
            trial_id,
            params=[Param(param_name, distribution.to_external_repr(param_value_internal))],
            tags=[RunTag(f"param_internal_val_{param_name}", json.dumps(param_value_internal))],
        )

        distributions_dict[param_name] = distribution_to_json(distribution)
        self._queue_batch_operation(
            trial_id, tags=[RunTag("param_directions", json.dumps(distributions_dict))]
        )

    def get_trial_id_from_study_id_trial_number(self, study_id, trial_number: int) -> int:
        raise NotImplementedError("This method is not supported in MLflow backend.")

    def get_trial_number_from_id(self, trial_id) -> int:
        # Flush the batch for this trial to ensure we have the latest data
        self._flush_batch(trial_id)

        trial_run = self._mlflow_client.get_run(trial_id)
        return int(trial_run.data.tags.get("numbers", 0))

    def get_trial_param(self, trial_id, param_name: str) -> float:
        # Flush the batch for this trial to ensure we have the latest data
        self._flush_batch(trial_id)

        trial_run = self._mlflow_client.get_run(trial_id)
        param_value = trial_run.data.tags[f"param_internal_val_{param_name}"]

        return float(json.loads(param_value))

    def set_trial_state_values(
        self, trial_id, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        # Update trial state
        if state.is_finished():
            self._mlflow_client.set_terminated(trial_id, status=optuna_mlflow_status_map[state])
        else:
            self._mlflow_client.update_run(trial_id, status=optuna_mlflow_status_map[state])

        # Queue value metrics if provided
        timestamp = int(time.time() * 1000)
        if values is not None:
            metrics = []
            if len(values) > 1:
                metrics = [
                    Metric(f"value_{idx}", val, timestamp, 1) for idx, val in enumerate(values)
                ]
            else:
                metrics = [Metric("value", values[0], timestamp, 1)]

            self._queue_batch_operation(trial_id, metrics=metrics)

        if state == TrialState.RUNNING and state != TrialState.WAITING:
            return False
        return True

    def set_trial_intermediate_value(self, trial_id, step: int, intermediate_value: float) -> None:
        # Queue intermediate value metric
        self._queue_batch_operation(
            trial_id,
            metrics=[
                Metric("intermediate_value", intermediate_value, int(time.time() * 1000), step)
            ],
        )

    def set_trial_user_attr(self, trial_id, key: str, value: Any) -> None:
        # Queue user attribute tag
        self._queue_batch_operation(trial_id, tags=[RunTag(f"user_{key}", json.dumps(value))])

    def set_trial_system_attr(self, trial_id, key: str, value: Any) -> None:
        # Queue system attribute tag
        self._queue_batch_operation(trial_id, tags=[RunTag(f"sys_{key}", json.dumps(value))])

    def get_trial(self, trial_id) -> FrozenTrial:
        # Flush the batch for this trial to ensure we have the latest data
        self._flush_batch(trial_id)

        trial_run = self._mlflow_client.get_run(trial_id)
        param_directions = trial_run.data.tags["param_directions"]
        try:
            distributions_dict = json.loads(param_directions)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(f"error with param_directions = {param_directions!r}") from e

        distributions = {
            k: json_to_distribution(distribution) for k, distribution in distributions_dict.items()
        }
        params = {}
        for key, value in trial_run.data.tags.items():
            if key.startswith("param_internal_val_"):
                param_name = key[19:]
                param_value = json.loads(value)
                params[param_name] = distributions[param_name].to_external_repr(float(param_value))

        metrics = trial_run.data.metrics
        values = None
        if "value" in metrics:
            values = [metrics["value"]]
        if "value_0" in metrics:
            values = [metrics[f"value_{idx}"] for idx in range(len(metrics))]

        run_number = int(trial_run.data.tags.get("numbers", 0))

        start_time = datetime.datetime.fromtimestamp(trial_run.info.start_time / 1000)
        if trial_run.info.end_time:
            end_time = datetime.datetime.fromtimestamp(trial_run.info.end_time / 1000)
        else:
            end_time = None
        return FrozenTrial(
            trial_id=trial_id,
            number=run_number,
            state=mlflow_optuna_status_map[trial_run.info.status],
            value=None,
            values=values,
            datetime_start=start_time,
            datetime_complete=end_time,
            params=params,
            distributions=distributions,
            user_attrs=self.get_trial_user_attrs(trial_id),
            system_attrs=self.get_trial_system_attrs(trial_id),
            intermediate_values={
                v.step: v.value
                for idx, v in enumerate(
                    self._mlflow_client.get_metric_history(trial_id, "intermediate_value")
                )
            },
        )

    def get_trial_user_attrs(self, trial_id) -> dict[str, Any]:
        # Flush the batch for this trial to ensure we have the latest data
        self._flush_batch(trial_id)

        run = self._mlflow_client.get_run(trial_id)
        user_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("user_"):
                user_attrs[key[5:]] = json.loads(value)
        return user_attrs

    def get_trial_system_attrs(self, trial_id) -> dict[str, Any]:
        # Flush the batch for this trial to ensure we have the latest data
        self._flush_batch(trial_id)

        run = self._mlflow_client.get_run(trial_id)
        system_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("sys_"):
                system_attrs[key[4:]] = json.loads(value)
        return system_attrs

    def get_all_trials(
        self,
        study_id,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        # Flush all batches to ensure we have the latest data
        self.flush_all_batches()

        runs = self._mlflow_client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=f"tags.mlflow.parentRunId='{study_id}'",
        )
        trials = []
        for run in runs:
            trials.append(self.get_trial(run.info.run_id))

        frozen_trials: list[FrozenTrial] = []
        for trial in trials:
            if states is None or trial.state in states:
                frozen_trials.append(trial)
        return frozen_trials

    def get_n_trials(self, study_id, states=None) -> int:
        # Flush all batches to ensure we have the latest data
        self.flush_all_batches()

        runs = self._mlflow_client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=f"tags.mlflow.parentRunId='{study_id}'",
        )
        return len(runs)
