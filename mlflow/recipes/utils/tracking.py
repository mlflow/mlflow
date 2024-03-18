import json
import logging
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional

import mlflow
from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.utils import get_recipe_name
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.tracking.fluent import set_experiment as fluent_set_experiment
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import path_to_local_file_uri, path_to_local_sqlite_uri
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
    LEGACY_MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_SOURCE_NAME,
)

_logger = logging.getLogger(__name__)


def _get_run_name(run_name_prefix):
    if run_name_prefix is None:
        return None

    sep = "-"
    num = uuid.uuid4().hex[:8]
    return f"{run_name_prefix}{sep}{num}"


class TrackingConfig:
    """
    The MLflow Tracking configuration associated with an MLflow Recipe, including the
    Tracking URI and information about the destination Experiment for writing results.
    """

    _KEY_TRACKING_URI = "mlflow_tracking_uri"
    _KEY_EXPERIMENT_NAME = "mlflow_experiment_name"
    _KEY_EXPERIMENT_ID = "mlflow_experiment_id"
    _KEY_RUN_NAME = "mlflow_run_name"
    _KEY_ARTIFACT_LOCATION = "mlflow_experiment_artifact_location"

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Args:
            tracking_uri: The MLflow Tracking URI.
            experiment_name: The MLflow Experiment name. At least one of ``experiment_name`` or
                ``experiment_id`` must be specified. If both are specified, they must be consistent
                with Tracking server state. Note that this Experiment may not exist prior to recipe
                execution.
            experiment_id: The MLflow Experiment ID. At least one of ``experiment_name`` or
                ``experiment_id`` must be specified. If both are specified, they must be consistent
                with Tracking server state. Note that this Experiment may not exist prior to recipe
                execution.
            run_name: The MLflow Run Name. If the run name is not specified, then a random name is
                set for the run.
            artifact_location: The artifact location to use for the Experiment, if the Experiment
                does not already exist. If the Experiment already exists, this location is ignored.
        """
        if tracking_uri is None:
            raise MlflowException(
                message="`tracking_uri` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if (experiment_name, experiment_id).count(None) != 1:
            raise MlflowException(
                message="Exactly one of `experiment_name` or `experiment_id` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.artifact_location = artifact_location

    def to_dict(self) -> Dict[str, str]:
        """
        Obtains a dictionary representation of the MLflow Tracking configuration.

        Returns:
            A dictionary representation of the MLflow Tracking configuration.
        """
        config_dict = {
            TrackingConfig._KEY_TRACKING_URI: self.tracking_uri,
        }

        if self.experiment_name:
            config_dict[TrackingConfig._KEY_EXPERIMENT_NAME] = self.experiment_name

        elif self.experiment_id:
            config_dict[TrackingConfig._KEY_EXPERIMENT_ID] = self.experiment_id

        if self.artifact_location:
            config_dict[TrackingConfig._KEY_ARTIFACT_LOCATION] = self.artifact_location

        if self.run_name:
            config_dict[TrackingConfig._KEY_RUN_NAME] = self.run_name

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, str]) -> "TrackingConfig":
        """
        Creates a ``TrackingConfig`` instance from a dictionary representation.

        Args:
            config_dict: A dictionary representation of the MLflow Tracking configuration.

        Returns:
            A ``TrackingConfig`` instance.
        """
        return TrackingConfig(
            tracking_uri=config_dict.get(TrackingConfig._KEY_TRACKING_URI),
            experiment_name=config_dict.get(TrackingConfig._KEY_EXPERIMENT_NAME),
            experiment_id=config_dict.get(TrackingConfig._KEY_EXPERIMENT_ID),
            run_name=config_dict.get(TrackingConfig._KEY_RUN_NAME),
            artifact_location=config_dict.get(TrackingConfig._KEY_ARTIFACT_LOCATION),
        )


def get_recipe_tracking_config(
    recipe_root_path: str, recipe_config: Dict[str, Any]
) -> TrackingConfig:
    """
    Obtains the MLflow Tracking configuration for the specified recipe.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.
        recipe_config: The configuration of the specified recipe.

    Returns:
        A ``TrackingConfig`` instance containing MLflow Tracking information for the
        specified recipe, including Tracking URI, Experiment name, and more.
    """
    if is_in_databricks_runtime():
        default_tracking_uri = "databricks"
        default_artifact_location = None
    else:
        mlflow_metadata_base_path = pathlib.Path(recipe_root_path) / "metadata" / "mlflow"
        mlflow_metadata_base_path.mkdir(exist_ok=True, parents=True)
        default_tracking_uri = path_to_local_sqlite_uri(
            path=str((mlflow_metadata_base_path / "mlruns.db").resolve())
        )
        default_artifact_location = path_to_local_file_uri(
            path=str((mlflow_metadata_base_path / "mlartifacts").resolve())
        )

    tracking_config = recipe_config.get("experiment", {})

    config_obj_kwargs = {
        "run_name": _get_run_name(tracking_config.get("run_name_prefix")),
        "tracking_uri": tracking_config.get("tracking_uri", default_tracking_uri),
        "artifact_location": tracking_config.get("artifact_location", default_artifact_location),
    }

    experiment_name = tracking_config.get("name")
    if experiment_name is not None:
        return TrackingConfig(
            experiment_name=experiment_name,
            **config_obj_kwargs,
        )

    experiment_id = tracking_config.get("id")
    if experiment_id is not None:
        return TrackingConfig(
            experiment_id=experiment_id,
            **config_obj_kwargs,
        )

    experiment_id = _get_experiment_id()
    if experiment_id != DEFAULT_EXPERIMENT_ID:
        return TrackingConfig(
            experiment_id=experiment_id,
            **config_obj_kwargs,
        )

    return TrackingConfig(
        experiment_name=get_recipe_name(recipe_root_path=recipe_root_path),
        **config_obj_kwargs,
    )


def apply_recipe_tracking_config(tracking_config: TrackingConfig):
    """
    Applies the specified ``TrackingConfig`` in the current context by setting the associated
    MLflow Tracking URI (via ``mlflow.set_tracking_uri()``) and setting the associated MLflow
    Experiment (via ``mlflow.set_experiment()``), creating it if necessary.

    Args:
        tracking_config: The MLflow Recipe ``TrackingConfig`` to apply.
    """
    mlflow.set_tracking_uri(uri=tracking_config.tracking_uri)

    client = MlflowClient()
    if tracking_config.experiment_name is not None:
        experiment = client.get_experiment_by_name(name=tracking_config.experiment_name)
        if not experiment:
            _logger.info(
                "Experiment with name '%s' does not exist. Creating a new experiment.",
                tracking_config.experiment_name,
            )
            try:
                client.create_experiment(
                    name=tracking_config.experiment_name,
                    artifact_location=tracking_config.artifact_location,
                )
            except RestException:
                # Inform user they should create an experiment and specify it in the recipe
                # config if an experiment with the recipe name can't be created.
                raise MlflowException(
                    f"Could not create an MLflow Experiment with "
                    f"name {tracking_config.experiment_name}. Please create an "
                    f"MLflow Experiment for this recipe and specify its name in the "
                    f'"name" field of the "experiment" section in your profile configuration.'
                )

    fluent_set_experiment(
        experiment_id=tracking_config.experiment_id, experiment_name=tracking_config.experiment_name
    )


def get_run_tags_env_vars(recipe_root_path: str) -> Dict[str, str]:
    """
    Returns environment variables that should be set during step execution to ensure that MLflow
    Run Tags from the current context are applied to any MLflow Runs that are created during
    recipe execution.

    Args:
        recipe_root_path: The absolute path of the recipe root directory on the local
            filesystem.

    Returns:
        A dictionary of environment variable names and values.
    """
    run_context_tags = resolve_tags()

    git_tags = {}
    git_repo_url = get_git_repo_url(path=recipe_root_path)
    if git_repo_url:
        git_tags[MLFLOW_SOURCE_NAME] = git_repo_url
        git_tags[MLFLOW_GIT_REPO_URL] = git_repo_url
        git_tags[LEGACY_MLFLOW_GIT_REPO_URL] = git_repo_url
    git_commit = get_git_commit(path=recipe_root_path)
    if git_commit:
        git_tags[MLFLOW_GIT_COMMIT] = git_commit
    git_branch = get_git_branch(path=recipe_root_path)
    if git_branch:
        git_tags[MLFLOW_GIT_BRANCH] = git_branch

    return {MLFLOW_RUN_CONTEXT.name: json.dumps({**run_context_tags, **git_tags})}


def log_code_snapshot(
    recipe_root: str,
    run_id: str,
    artifact_path: str = "recipe_snapshot",
    recipe_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Logs a recipe code snapshot as mlflow artifacts.

    Args:
        recipe_root_path: String file path to the directory where the recipe is defined.
        run_id: Run ID to which the code snapshot is logged.
        artifact_path: Directory within the run's artifact director (default: "snapshots").
        recipe_config: Dict containing the full recipe configuration at runtime.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        recipe_root = pathlib.Path(recipe_root)
        for file_path in (
            # TODO: Log a filled recipe.yaml created in `Recipe._resolve_recipe_steps`
            #       instead of a raw recipe.yaml.
            recipe_root.joinpath("recipe.yaml"),
            recipe_root.joinpath("requirements.txt"),
            *recipe_root.glob("profiles/*.yaml"),
            *recipe_root.glob("steps/*.py"),
        ):
            if file_path.exists():
                tmp_path = tmpdir.joinpath(file_path.relative_to(recipe_root))
                tmp_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(file_path, tmp_path)
        if recipe_config is not None:
            import yaml

            tmp_path = tmpdir.joinpath("runtime/recipe.yaml")
            tmp_path.parent.mkdir(exist_ok=True, parents=True)
            with open(tmp_path, mode="w", encoding="utf-8") as config_file:
                yaml.dump(recipe_config, config_file)
        MlflowClient().log_artifacts(run_id, str(tmpdir), artifact_path=artifact_path)
