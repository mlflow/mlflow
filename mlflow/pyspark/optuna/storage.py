from collections.abc import Container
from collections.abc import Sequence

import copy
import datetime
import json
import sys
import time
from typing import Any, Optional, List, Dict
import uuid

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

try:
    from optuna._typing import JSONSerializable
    from optuna.distributions import BaseDistribution
    from optuna.distributions import distribution_to_json
    from optuna.distributions import json_to_distribution, check_distribution_compatibility
    from optuna.storages import BaseStorage
    from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
    from optuna.study import StudyDirection
    from optuna.study._frozen import FrozenStudy
    from optuna.trial import FrozenTrial
    from optuna.trial import TrialState
except ImportError:
    sys.exit()

optuna_mlflow_status_map = {
    TrialState.RUNNING: "RUNNING",
    TrialState.COMPLETE: "FINISHED",
    TrialState.PRUNED: "KILLED",
    TrialState.FAIL: "FAILED",
    TrialState.WAITING: "SCHEDULED"
}

mlflow_optuna_status_map = {
    "RUNNING": TrialState.RUNNING,
    "FINISHED": TrialState.COMPLETE,
    "KILLED": TrialState.PRUNED,
    "FAILED": TrialState.FAIL,
    "SCHEDULED": TrialState.WAITING
}


class MLFlowStorage(BaseStorage):
    """
    MLFlow based storage class.
    """

    def __init__(self, experiment_id: str, name: Optional[str] = None):
        if not experiment_id:
            raise Exception(
                "MLFlowStorage need import & save results from the experimens. No experiment_id is provided"
            )

        self._experiment_id = experiment_id
        self._mlflow_client = MlflowClient()
        self._name = name

    def _search_run_by_name(self, run_name: str):
        # print(f"Searching run by name: {run_name}")
        filter_string = f"tags.`mlflow.runName` = '{run_name}'"
        runs = self._mlflow_client.search_runs(
            experiment_ids=[self._experiment_id], filter_string=filter_string)
        return runs

    def create_new_study(self,
                         directions: Sequence[StudyDirection],
                         study_name: Optional[str] = None) -> int:
        """Create a new study as a mlflow run.
        Parameters
        ----------
        directions: A sequence of direction whose element is either
                 :obj:`~optuna.study.StudyDirection.MAXIMIZE` or
                 :obj:`~optuna.study.StudyDirection.MINIMIZE`.
        study_name: str, optional
                Name of the new study to create.
        Returns
        ------
        str
        run ID of the created study mlflow run.
        """
        study_name = study_name or DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4())
        tags = {
            "mlflow.runName": study_name,
            "optuna.study_direction": ",".join(direction.name for direction in directions)
        }
        study_run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=tags)

        study_id = study_run.info.run_id
        return study_id

    def delete_study(self, study_id) -> None:
        """Delete a study.
        Parameters
        ----------
        study_id: str
            mlflow run ID of the study.
        """
        self._mlflow_client.delete_run(study_id)

    def set_study_user_attr(self, study_id, key: str, value: JSONSerializable) -> None:
        """Register a user-defined attribute as mlflow run tags to a study run.
        This method overwrites any existing attribute.
        Parameters
        ----------
        study_id: str
            mlflow run ID of the study.
        key: str
            Attribute key.
        value: JSONSerializable
            Attribute value. It should be JSON serializable.
        """
        self._mlflow_client.set_tag(study_id, f"user_{key}", json.dumps(value))

    def set_study_system_attr(self, study_id, key: str, value: JSONSerializable) -> None:
        """Register a optuna-internal attribute as mlflow run tags to a study run.
        This method overwrites any existing attribute.
        Parameters
        ----------
        study_id: str
            mlflow run ID of the study.
        key: str
            Attribute key.
        value: JSONSerializable
            Attribute value. It should be JSON serializable.
        """
        self._mlflow_client.set_tag(study_id, f"sys_{key}", json.dumps(value))

    def get_study_id_from_name(self, study_name: str) -> int:
        runs = self._search_run_by_name(study_name)
        if len(runs):
            return runs[0].info.run_id
        else:
            raise Exception(f"Study {study_name} not found")

    def get_study_name_from_id(self, study_id) -> str:
        run = self._mlflow_client.get_run(study_id)
        return run.data.tags["mlflow.runName"]

    def get_study_directions(self, study_id) -> List[StudyDirection]:
        run = self._mlflow_client.get_run(study_id)
        directions_str = run.data.tags["optuna.study_direction"]
        directions = [StudyDirection[name] for name in directions_str.split(",")]
        return directions

    def get_study_user_attrs(self, study_id) -> Dict[str, Any]:
        run = self._mlflow_client.get_run(study_id)
        user_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("user_"):
                user_attrs[key[5:]] = json.loads(value)
        return user_attrs

    def get_study_system_attrs(self, study_id) -> Dict[str, Any]:
        run = self._mlflow_client.get_run(study_id)
        system_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("sys_"):
                system_attrs[key[4:]] = json.loads(value)
        return system_attrs

    def get_all_studies(self) -> List[FrozenStudy]:
        # print("Getting all studies")
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
                ))
        return studies

    def create_new_trial(self, study_id, template_trial: Optional[FrozenTrial] = None) -> int:
        # print(f"Creating new trial for study ID: {study_id}")
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
            k: distribution_to_json(dist)
            for k, dist in frozen.distributions.items()
        }
        distribution_str = json.dumps(distribution_json)
        tags = {"param_directions": distribution_str}

        trial_run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=tags)
        trial_id = trial_run.info.run_id
        self._mlflow_client.set_tag(trial_id, MLFLOW_PARENT_RUN_ID, study_id)
        hash_id = float(hash(trial_id))
        self._mlflow_client.log_metric(study_id, "trial_id", hash_id)
        trial_ids = self._mlflow_client.get_metric_history(study_id, "trial_id")
        index = next((i for i, obj in enumerate(trial_ids) if obj.value == hash_id), -1)
        self._mlflow_client.set_tag(trial_id, "numbers", index)

        state = frozen.state
        if state.is_finished():
            self._mlflow_client.set_terminated(trial_id, status=optuna_mlflow_status_map[state])
        else:
            self._mlflow_client.update_run(trial_id, status=optuna_mlflow_status_map[state])

        timestamp = int(time.time() * 1000)
        if frozen.values is not None:
            if len(frozen.values) > 1:
                metrics = [
                    Metric(f"value_{idx}", val, timestamp, 1)
                    for idx, val in enumerate(frozen.values)
                ]
            else:
                metrics = [Metric("value", frozen.values[0], timestamp, 1)]
        elif frozen.value is not None:
            metrics = [Metric("value", frozen.value, timestamp, 1)]
        else:
            metrics = []

        params = [Param(k, param) for k, param in frozen.params.items()]
        tags = [
                   RunTag(f"user_{key}", json.dumps(value)) for key, value in frozen.user_attrs.items()
               ] + [RunTag(f"sys_{key}", json.dumps(value))
                    for key, value in frozen.system_attrs.items()] + [
                   RunTag(f"param_internal_val_{k}",
                          json.dumps(frozen.distributions[k].to_internal_repr(param)))
                   for k, param in frozen.params.items()
               ]

        metrics = metrics + [
            Metric("intermediate_value", val, timestamp, int(k))
            for k, val in frozen.intermediate_values.items()
        ]
        self._mlflow_client.log_batch(trial_id, params=params, metrics=metrics, tags=tags)

        return trial_id

    def set_trial_param(
            self,
            trial_id,
            param_name: str,
            param_value_internal: float,
            distribution: BaseDistribution,
    ) -> None:
        trial_run = self._mlflow_client.get_run(trial_id)
        distributions_dict = json.loads(trial_run.data.tags["param_directions"])
        self.check_trial_is_updatable(trial_id, mlflow_optuna_status_map[trial_run.info.status])

        if param_name in trial_run.data.params:
            param_distribution = json_to_distribution(distributions_dict[param_name])
            check_distribution_compatibility(param_distribution, distribution)

        self._mlflow_client.log_param(trial_id, param_name,
                                      distribution.to_external_repr(param_value_internal))
        self._mlflow_client.set_tag(trial_id, f"param_internal_val_{param_name}",
                                    param_value_internal)

        distributions_dict[param_name] = distribution_to_json(distribution)
        self._mlflow_client.set_tag(trial_id, "param_directions", json.dumps(distributions_dict))

    def get_trial_id_from_study_id_trial_number(self, study_id, trial_number: int) -> int:
        raise NotImplementedError("This method is not supported in MLflow backend.")

    def get_trial_number_from_id(self, trial_id) -> int:
        trial_run = self._mlflow_client.get_run(trial_id)
        return int(trial_run.data.tags["numbers"])

    def get_trial_param(self, trial_id, param_name: str) -> float:
        trial_run = self._mlflow_client.get_run(trial_id)
        param_value = trial_run.data.tags[f"param_internal_val_{param_name}"]

        return float(param_value)

    def set_trial_state_values(self,
                               trial_id,
                               state: TrialState,
                               values: Optional[Sequence[float]] = None) -> bool:
        if state.is_finished():
            self._mlflow_client.set_terminated(trial_id, status=optuna_mlflow_status_map[state])
        else:
            self._mlflow_client.update_run(trial_id, status=optuna_mlflow_status_map[state])

        timestamp = int(time.time() * 1000)
        if values is not None:
            if len(values) > 1:
                metrics = [
                    Metric(f"value_{idx}", val, timestamp, 1) for idx, val in enumerate(values)
                ]
            else:
                metrics = [Metric("value", values[0], timestamp, 1)]
        else:
            metrics = []
        self._mlflow_client.log_batch(trial_id, metrics=metrics)

        if state == TrialState.RUNNING and state != TrialState.WAITING:
            return False
        return True

    def set_trial_intermediate_value(self, trial_id, step: int, intermediate_value: float) -> None:
        self._mlflow_client.log_metric(
            trial_id, "intermediate_value", intermediate_value, step=step)

    def set_trial_user_attr(self, trial_id, key: str, value: Any) -> None:
        self._mlflow_client.set_tag(trial_id, f"user_{key}", json.dumps(value))

    def set_trial_system_attr(self, trial_id, key: str, value: Any) -> None:
        self._mlflow_client.set_tag(trial_id, f"sys_{key}", json.dumps(value))

    def get_trial(self, trial_id) -> FrozenTrial:
        trial_run = self._mlflow_client.get_run(trial_id)
        distributions_dict = json.loads(trial_run.data.tags["param_directions"])
        distributions = {
            k: json_to_distribution(distribution)
            for k, distribution in distributions_dict.items()
        }
        params = {}
        for key, value in trial_run.data.tags.items():
            if key.startswith("param_internal_val_"):
                param_name = key[19:]
                params[param_name] = distributions[param_name].to_external_repr(float(value))

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
                    self._mlflow_client.get_metric_history(trial_id, "intermediate_value"))
            },
        )

    def get_trial_user_attrs(self, trial_id) -> Dict[str, Any]:
        run = self._mlflow_client.get_run(trial_id)
        user_attrs = {}
        for key, value in run.data.tags.items():
            if key.startswith("user_"):
                user_attrs[key[5:]] = json.loads(value)
        return user_attrs

    def get_trial_system_attrs(self, trial_id) -> Dict[str, Any]:
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
            states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        runs = self._mlflow_client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=f"tags.mlflow.parentRunId='{study_id}'")
        trials = []
        for run in runs:
            trials.append(self.get_trial(run.info.run_id))

        frozen_trials: List[FrozenTrial] = []
        for trial in trials:
            if states is None or trial.state in states:
                frozen_trials.append(trial)
        return frozen_trials

    def get_n_trials(self, study_id, states=None) -> int:
        runs = self._mlflow_client.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=f"tags.mlflow.parentRunId='{study_id}'")
        return len(runs)
