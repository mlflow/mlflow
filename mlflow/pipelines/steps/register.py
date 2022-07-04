import json
import logging
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

_logger = logging.getLogger(__name__)


class RegisterStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super(RegisterStep, self).__init__(step_config, pipeline_root)
        self.tracking_config = TrackingConfig.from_dict(step_config)
        self.num_dropped_rows = None
        self.model_uri = None
        self.model_details = None
        self.alerts = None
        self.version = None

        if "model_name" not in self.step_config:
            raise MlflowException(
                "Missing 'model_name' config in register step config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.register_model_name = self.step_config["model_name"]
        self.allow_non_validated_model = self.step_config.get("allow_non_validated_model", False)

    def _run(self, output_directory):
        apply_pipeline_tracking_config(self.tracking_config)

        run_id_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="train",
            relative_path="run_id",
        )
        run_id = Path(run_id_path).read_text()

        model_validation_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="evaluate",
            relative_path="model_validation_status",
        )
        model_validation = Path(model_validation_path).read_text()
        artifact_path = "train/model"
        if model_validation == "VALIDATED" or (
            model_validation == "UNKNOWN" and self.allow_non_validated_model
        ):
            self.model_uri = "runs:/{run_id}/{artifact_path}".format(
                run_id=run_id, artifact_path=artifact_path
            )
            self.model_details = mlflow.register_model(
                model_uri=self.model_uri,
                name=self.register_model_name,
                await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
            )
            self.version = self.model_details.version
            registered_model_info = RegisteredModelVersionInfo(
                name=self.register_model_name, version=self.version
            )
            registered_model_info.to_json(
                path=str(Path(output_directory) / "registered_model_version.json")
            )
        else:
            self.alerts = (
                "Model registration skipped.  Please check the validation "
                "result from Evaluate step."
            )

        card = self._build_card()
        card.save_as_html(output_directory)
        self._log_step_card(run_id, self.name)
        return card

    def _build_card(self) -> BaseCard:
        # Build card

        final_markdown = []
        if self.model_uri is not None:
            final_markdown.append(f"**Model Source URI:** `{self.model_uri}`")
        if self.version is not None:
            final_markdown.append(f"**Model Name:** `{self.register_model_name}`")
        if self.version is not None:
            final_markdown.append(f"**Model Version:** `{self.version}`")
        if self.alerts is not None:
            final_markdown.append(f"**Alerts:** `{self.alerts}`")
        card = BaseCard(self.pipeline_name, self.name)
        card.add_tab(
            "Run Summary", "{{ SUMMARY }}" + "{{ EXE_DURATION }}" + "{{ LAST_UPDATE_TIME }}"
        ).add_markdown("SUMMARY", "<br>\n".join(final_markdown))
        return card

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["register"]
            step_config.update(
                get_pipeline_tracking_config(
                    pipeline_root_path=pipeline_root,
                    pipeline_config=pipeline_config,
                ).to_dict()
            )
        except KeyError:
            raise MlflowException(
                "Config for register step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "register"

    @property
    def environment(self):
        return get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)


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
        with open(path, "r") as f:
            registered_model_info_dict = json.load(f)

        return cls(
            name=registered_model_info_dict[RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_NAME],
            version=registered_model_info_dict[
                RegisteredModelVersionInfo._KEY_REGISTERED_MODEL_VERSION
            ],
        )
