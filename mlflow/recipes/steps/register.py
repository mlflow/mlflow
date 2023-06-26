import logging
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow.entities import SourceType
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.recipes.artifacts import ModelVersionArtifact, RegisteredModelVersionInfo
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep
from mlflow.recipes.step import StepClass
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.tracking import (
    get_recipe_tracking_config,
    apply_recipe_tracking_config,
    TrackingConfig,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.databricks_utils import (
    get_databricks_env_vars,
    get_databricks_model_version_url,
    get_databricks_run_url,
)
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_TYPE, MLFLOW_RECIPE_TEMPLATE_NAME

_logger = logging.getLogger(__name__)

_REGISTERED_MV_INFO_FILE = "registered_model_version.json"


class RegisterStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], recipe_root: str):
        super().__init__(step_config, recipe_root)
        self.tracking_config = TrackingConfig.from_dict(self.step_config)

    def _validate_and_apply_step_config(self):
        self.num_dropped_rows = None
        self.model_uri = None
        self.model_details = None
        self.version = None

        self.register_model_name = self.step_config.get("model_name")
        if self.register_model_name is None:
            raise MlflowException(
                "Missing 'model_name' config in register step config.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self.allow_non_validated_model = self.step_config.get("allow_non_validated_model", False)
        self.registry_uri = self.step_config.get("registry_uri", None)

    def _run(self, output_directory):
        apply_recipe_tracking_config(self.tracking_config)

        run_id_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
            step_name="train",
            relative_path="run_id",
        )
        run_id = Path(run_id_path).read_text()

        model_validation_path = get_step_output_path(
            recipe_root_path=self.recipe_root,
            step_name="evaluate",
            relative_path="model_validation_status",
        )
        model_validation = Path(model_validation_path).read_text()
        artifact_path = "train/model"
        tags = {
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.RECIPE),
            MLFLOW_RECIPE_TEMPLATE_NAME: self.step_config["recipe"],
        }
        self.model_uri = f"runs:/{run_id}/{artifact_path}"
        if model_validation == "VALIDATED" or (
            model_validation == "UNKNOWN" and self.allow_non_validated_model
        ):
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)
            self.model_details = mlflow.register_model(
                model_uri=self.model_uri,
                name=self.register_model_name,
                tags=tags,
                await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
            )
            self.version = self.model_details.version
            registered_model_info = RegisteredModelVersionInfo(
                name=self.register_model_name, version=self.version
            )
            registered_model_info.to_json(
                path=str(Path(output_directory) / _REGISTERED_MV_INFO_FILE)
            )
        else:
            raise MlflowException(
                f"Model registration on {self.model_uri} failed because it "
                "is not validated. Bypass by setting allow_non_validated_model to True. "
            )

        card = self._build_card(run_id)
        card.save_as_html(output_directory)
        self._log_step_card(run_id, self.name)
        return card

    def _build_card(self, run_id: str) -> BaseCard:
        card = BaseCard(self.recipe_name, self.name)
        card_tab = card.add_tab(
            "Run Summary",
            "{{ MODEL_NAME }}"
            + "{{ MODEL_VERSION }}"
            + "{{ MODEL_SOURCE_URI }}"
            + "{{ ALERTS }}"
            + "{{ EXE_DURATION }}"
            + "{{ LAST_UPDATE_TIME }}",
        )

        if self.version is not None:
            model_version_url = get_databricks_model_version_url(
                registry_uri=mlflow.get_registry_uri(),
                name=self.register_model_name,
                version=self.version,
            )

            if model_version_url is not None:
                card_tab.add_html(
                    "MODEL_NAME",
                    (
                        f"<b>Model Name:</b> <a href={model_version_url}>"
                        f"{self.register_model_name}</a><br><br>"
                    ),
                )
                card_tab.add_html(
                    "MODEL_VERSION",
                    (
                        f"<b>Model Version</b> <a href={model_version_url}>"
                        f"{self.version}</a><br><br>"
                    ),
                )
            else:
                card_tab.add_markdown(
                    "MODEL_NAME",
                    f"**Model Name:** `{self.register_model_name}`",
                )
                card_tab.add_markdown(
                    "MODEL_VERSION",
                    f"**Model Version:** `{self.version}`",
                )

        model_source_url = get_databricks_run_url(
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run_id,
            artifact_path=f"train/{TrainStep.MODEL_ARTIFACT_RELATIVE_PATH}",
        )

        if self.model_uri is not None and model_source_url is not None:
            card_tab.add_html(
                "MODEL_SOURCE_URI",
                (f"<b>Model Source URI</b> <a href={model_source_url}>" f"{self.model_uri}</a>"),
            )
        elif self.model_uri is not None:
            card_tab.add_markdown(
                "MODEL_SOURCE_URI",
                f"**Model Source URI:** `{self.model_uri}`",
            )

        return card

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get("steps", {}).get("register") is not None:
            step_config.update(recipe_config.get("steps", {}).get("register"))
        step_config["recipe"] = recipe_config.get("recipe")
        if recipe_config.get("model_registry", {}).get("registry_uri") is not None:
            step_config["registry_uri"] = recipe_config.get("model_registry", {}).get(
                "registry_uri"
            )
        if recipe_config.get("model_registry", {}).get("model_name") is not None:
            step_config["model_name"] = recipe_config.get("model_registry", {}).get("model_name")
        step_config.update(
            get_recipe_tracking_config(
                recipe_root_path=recipe_root,
                recipe_config=recipe_config,
            ).to_dict()
        )
        return cls(step_config, recipe_root)

    @property
    def name(self):
        return "register"

    @property
    def environment(self):
        return get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)

    def get_artifacts(self):
        return [
            ModelVersionArtifact(
                "registered_model_version",
                self.recipe_root,
                self.name,
                self.tracking_config.tracking_uri,
            )
        ]

    def step_class(self):
        return StepClass.TRAINING
