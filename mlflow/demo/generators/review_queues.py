from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas import InputCategorical, InputPassFail, InputText
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

_DEMO_CREATED_BY = "demo"

# Custom queue bundling a curated set of traces and questions for an expert review pass.
DEMO_REVIEW_QUEUE_NAME = "Demo Response Review"
# Reviewers assigned to the custom queue. Each also gets a personal user queue so the
# demo shows both queue flavors (a curated custom task and per-reviewer worklists).
DEMO_REVIEWERS = ["alice", "bob"]
# The no-auth current viewer. Its user queue is the "Default" queue the UI shows, so we
# seed it too (it would otherwise render empty until the viewer flags traces themselves).
DEMO_DEFAULT_REVIEWER = "default"

# Questions (label schemas) reviewers answer for each attached trace.
DEMO_LABEL_SCHEMAS = [
    {
        "name": "response_quality",
        "type": "feedback",
        "input": InputCategorical(options=["Excellent", "Good", "Fair", "Poor"]),
        "instruction": "Rate the overall quality of the assistant's response.",
        "enable_comment": True,
    },
    {
        "name": "is_helpful",
        "type": "feedback",
        "input": InputPassFail(positive_label="Helpful", negative_label="Not helpful"),
        "instruction": "Did the response help the user accomplish their goal?",
        "enable_comment": False,
    },
    {
        "name": "correct_answer",
        "type": "expectation",
        "input": InputText(),
        "instruction": "If the response was wrong, provide the correct answer.",
        "enable_comment": False,
    },
]

_MAX_QUEUE_ITEMS = 12
_MAX_USER_QUEUE_ITEMS = 5


class ReviewQueuesDemoGenerator(BaseDemoGenerator):
    """Generates demo review queues showing the expert trace-review workflow.

    Creates a set of label schemas (the questions reviewers answer), one curated
    ``custom`` queue with traces attached and mixed review progress, and a personal
    ``user`` queue per demo reviewer. Together these populate the Review Queue UI
    with a realistic worklist so users can explore the labeling experience.
    """

    name = DemoFeature.REVIEW_QUEUES
    version = 1

    def generate(self) -> DemoResult:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found")

        experiment_id = experiment.experiment_id

        schema_ids = [
            self._get_or_create_label_schema(store, experiment_id, spec).schema_id
            for spec in DEMO_LABEL_SCHEMAS
        ]

        traces = mlflow.search_traces(
            locations=[experiment_id], max_results=1000, return_type="list", flush=True
        )
        trace_ids = [trace.info.trace_id for trace in traces]

        custom_queue = store.create_review_queue(
            experiment_id,
            name=DEMO_REVIEW_QUEUE_NAME,
            queue_type="custom",
            created_by=_DEMO_CREATED_BY,
            users=DEMO_REVIEWERS,
            schema_ids=schema_ids,
        )
        created_queue_ids = [custom_queue.queue_id]

        if custom_item_ids := trace_ids[:_MAX_QUEUE_ITEMS]:
            store.add_items_to_review_queue(custom_queue.queue_id, item_ids=custom_item_ids)
            self._seed_progress(store, custom_queue.queue_id, custom_item_ids)

        # Personal user queues: each reviewer's own worklist over a few traces. Includes
        # the default (viewer) queue so the "Default" tab is populated out of the box.
        for offset, reviewer in enumerate([*DEMO_REVIEWERS, DEMO_DEFAULT_REVIEWER]):
            user_queue = store.get_or_create_user_queue(experiment_id, user=reviewer)
            created_queue_ids.append(user_queue.queue_id)
            start = offset * _MAX_USER_QUEUE_ITEMS
            if user_item_ids := trace_ids[start : start + _MAX_USER_QUEUE_ITEMS]:
                store.add_items_to_review_queue(user_queue.queue_id, item_ids=user_item_ids)

        return DemoResult(
            feature=self.name,
            entity_ids=created_queue_ids,
            navigation_url=f"#/experiments/{experiment_id}/review-queue",
        )

    def _get_or_create_label_schema(self, store, experiment_id: str, spec: dict[str, Any]):
        """Create a demo label schema, reusing an existing one with the same name.

        Idempotent so a re-run after a partially-completed generation (schemas created
        but queues not, which `_data_exists` doesn't detect) doesn't crash on the
        duplicate-name ``RESOURCE_ALREADY_EXISTS`` from ``create_label_schema``.
        """
        try:
            return store.create_label_schema(
                experiment_id,
                name=spec["name"],
                type=spec["type"],
                input=spec["input"],
                instruction=spec["instruction"],
                enable_comment=spec["enable_comment"],
            )
        except MlflowException as e:
            if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise
            return store.get_label_schema_by_name(experiment_id, spec["name"])

    def _seed_progress(self, store, queue_id: str, item_ids: list[str]) -> None:
        """Mark a mix of items complete/declined so the queue shows realistic progress."""
        for index, item_id in enumerate(item_ids):
            reviewer = DEMO_REVIEWERS[index % len(DEMO_REVIEWERS)]
            # Roughly: every 3rd item complete, every 5th declined, rest left pending.
            if index % 5 == 4:
                status = "declined"
            elif index % 3 == 0:
                status = "complete"
            else:
                continue
            store.set_review_queue_item_status(
                queue_id, item_id=item_id, status=status, completed_by=reviewer
            )

    def _data_exists(self) -> bool:
        try:
            store = _get_store()
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None:
                return False
            store.get_review_queue_by_name(experiment.experiment_id, name=DEMO_REVIEW_QUEUE_NAME)
            return True
        except Exception:
            return False

    def delete_demo(self) -> None:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            return

        experiment_id = experiment.experiment_id
        for name in [DEMO_REVIEW_QUEUE_NAME, *DEMO_REVIEWERS, DEMO_DEFAULT_REVIEWER]:
            try:
                queue = store.get_review_queue_by_name(experiment_id, name=name)
                store.delete_review_queue(queue.queue_id)
            except Exception:
                _logger.debug("Failed to delete demo review queue %s", name, exc_info=True)

        # Look each demo schema up by name rather than paging through the experiment's
        # schemas: direct lookup can't miss a demo schema regardless of how many
        # (possibly user-created) schemas the experiment has.
        for spec in DEMO_LABEL_SCHEMAS:
            try:
                schema = store.get_label_schema_by_name(experiment_id, spec["name"])
                if not schema.is_default:
                    store.delete_label_schema(schema.schema_id)
            except Exception:
                _logger.debug("Failed to delete demo label schema %s", spec["name"], exc_info=True)
