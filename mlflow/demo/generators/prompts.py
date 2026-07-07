from __future__ import annotations

import logging

from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    DEMO_PROMPT_PREFIX,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.demo.data import DEMO_PROMPTS, DemoPromptDef
from mlflow.genai.prompts import (
    delete_prompt_alias,
    register_prompt,
    search_prompts,
    set_prompt_alias,
)
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)


class PromptsDemoGenerator(BaseDemoGenerator):
    """Generates demo prompts showing version history and alias management.

    Creates:
    - 3 prompts: customer-support, document-summarizer, code-reviewer
    - Each with 3-4 versions showing prompt evolution
    - Version-specific aliases (baseline, improvements, production)
    """

    name = DemoFeature.PROMPTS
    version = 1

    def generate(self) -> DemoResult:
        import mlflow

        self._restore_experiment_if_deleted()
        mlflow.set_experiment(DEMO_EXPERIMENT_NAME)

        prompt_names = []
        total_versions = 0

        for prompt_def in DEMO_PROMPTS:
            versions_created = self._create_prompt_with_versions(prompt_def)
            prompt_names.append(prompt_def.name)
            total_versions += versions_created

        entity_ids = [
            f"prompts:{len(prompt_names)}",
            f"versions:{total_versions}",
        ]

        return DemoResult(
            feature=self.name,
            entity_ids=entity_ids,
            navigation_url="#/prompts",
        )

    def _create_prompt_with_versions(self, prompt_def: DemoPromptDef) -> int:
        for version_num, version_def in enumerate(prompt_def.versions, start=1):
            register_prompt(
                name=prompt_def.name,
                template=version_def.template,
                commit_message=version_def.commit_message,
                tags={"demo": "true"},
            )

            if version_def.aliases:
                set_prompt_alias(
                    name=prompt_def.name,
                    alias=version_def.aliases[0],
                    version=version_num,
                )

        return len(prompt_def.versions)

    def _data_exists(self) -> bool:
        try:
            prompts = search_prompts(
                filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
                max_results=1,
            )
            return len(prompts) > 0
        except Exception:
            _logger.debug("Failed to check if prompts demo exists", exc_info=True)
            return False

    def delete_demo(self) -> None:
        all_aliases = set()
        for prompt_def in DEMO_PROMPTS:
            for version_def in prompt_def.versions:
                all_aliases.update(version_def.aliases)

        try:
            prompts = search_prompts(
                filter_string=f"name LIKE '{DEMO_PROMPT_PREFIX}.%'",
                max_results=100,
            )

            client = MlflowClient()
            for prompt in prompts:
                try:
                    for alias in all_aliases:
                        try:
                            delete_prompt_alias(name=prompt.name, alias=alias)
                        except Exception:
                            _logger.debug(
                                "Failed to delete alias %s for prompt %s",
                                alias,
                                prompt.name,
                                exc_info=True,
                            )
                    client.delete_prompt(name=prompt.name)
                except Exception:
                    _logger.debug("Failed to delete prompt %s", prompt.name, exc_info=True)
        except Exception:
            _logger.debug("Failed to delete demo prompts", exc_info=True)

    def _restore_experiment_if_deleted(self) -> None:
        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is not None and experiment.lifecycle_stage == "deleted":
                _logger.info("Restoring soft-deleted demo experiment")
                client = MlflowClient()
                client.restore_experiment(experiment.experiment_id)
        except Exception:
            _logger.debug("Failed to check/restore demo experiment", exc_info=True)
