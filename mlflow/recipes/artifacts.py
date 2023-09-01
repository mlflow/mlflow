import json
import logging
import os
from abc import ABC, abstractmethod

import mlflow
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import chdir

_logger = logging.getLogger(__name__)


class Artifact(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def path(self):
        pass

    @abstractmethod
    def load(self):
        pass


class DataframeArtifact(Artifact):
    def __init__(self, name, recipe_root, step_name, rel_path=""):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, rel_path)
        self._step_name = step_name

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        import pandas as pd

        if os.path.exists(self._path):
            return pd.read_parquet(self._path)
        log_artifact_not_found_warning(self._name, self._step_name)
        return None


class ModelArtifact(Artifact):
    def __init__(self, name, recipe_root, step_name, tracking_uri):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, "model/model.pkl")
        self._recipe_root = recipe_root
        self._step_name = step_name
        self._tracking_uri = tracking_uri

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        run_id = read_run_id(self._recipe_root)
        if run_id:
            with _use_tracking_uri(self._tracking_uri), chdir(self._recipe_root):
                return mlflow.pyfunc.load_model(f"runs:/{run_id}/{self._step_name}/model")
        log_artifact_not_found_warning(self._name, self._step_name)
        return None


class TransformerArtifact(Artifact):
    def __init__(self, name, recipe_root, step_name, tracking_uri):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, "transformer.pkl")
        self._recipe_root = recipe_root
        self._step_name = step_name
        self._tracking_uri = tracking_uri

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        run_id = read_run_id(self._recipe_root)
        if run_id:
            with _use_tracking_uri(self._tracking_uri), chdir(self._recipe_root):
                return mlflow.sklearn.load_model(f"runs:/{run_id}/{self._step_name}/transformer")
        log_artifact_not_found_warning(self._name, self._step_name)
        return None


class RunArtifact(Artifact):
    def __init__(self, name, recipe_root, step_name, tracking_uri):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, "run_id")
        self._recipe_root = recipe_root
        self._step_name = step_name
        self._tracking_uri = tracking_uri

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        run_id = read_run_id(self._recipe_root)
        if run_id:
            with _use_tracking_uri(self._tracking_uri), chdir(self._recipe_root):
                return MlflowClient().get_run(run_id)
        log_artifact_not_found_warning(self._name, self._step_name)
        return None


class ModelVersionArtifact(Artifact):
    def __init__(self, name, recipe_root, step_name, tracking_uri):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, "registered_model_version.json")
        self._recipe_root = recipe_root
        self._step_name = step_name
        self._tracking_uri = tracking_uri

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        if os.path.exists(self._path):
            registered_model_info = RegisteredModelVersionInfo.from_json(path=self._path)
            with _use_tracking_uri(self._tracking_uri), chdir(self._recipe_root):
                return MlflowClient().get_model_version(
                    name=registered_model_info.name, version=registered_model_info.version
                )
        log_artifact_not_found_warning(self._name, self._step_name)
        return None


class HyperParametersArtifact(Artifact):
    def __init__(self, name, recipe_root, step_name):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, "best_parameters.yaml")

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        if os.path.exists(self._path):
            return open(self._path).read()


def log_artifact_not_found_warning(artifact_name, step_name):
    _logger.warning(
        f"The artifact with name '{artifact_name}' was not found."
        f" Re-run the '{step_name}' step to generate it."
    )


def read_run_id(recipe_root):
    run_id_file_path = get_step_output_path(recipe_root, "train", "run_id")
    if os.path.exists(run_id_file_path):
        with open(run_id_file_path) as f:
            return f.read().strip()
    return None


class RegisteredModelVersionInfo:
    _KEY_REGISTERED_MODEL_NAME = "registered_model_name"
    _KEY_REGISTERED_MODEL_VERSION = "registered_model_version"

    def __init__(self, name: str, version: int):
        self.name = name
        self.version = version

    def to_json(self, path):
        registered_model_info_dict = {
            RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_NAME: self.name,
            RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_VERSION: self.version,
        }
        with open(path, "w") as f:
            json.dump(registered_model_info_dict, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            registered_model_info_dict = json.load(f)

        return cls(
            name=registered_model_info_dict[RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_NAME],
            version=registered_model_info_dict[
                RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_VERSION
            ],
        )
