from __future__ import annotations

import json
import logging

from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_CUSTOM_VIEW_TAG_PREFIX
from mlflow.utils.validation import MAX_EXPERIMENT_TAG_VAL_LENGTH

_logger = logging.getLogger(__name__)

DEMO_CUSTOM_VIEW_ID = "mlflow-demo-span-review"
DEMO_CUSTOM_VIEW_TAG_KEY = f"{MLFLOW_CUSTOM_VIEW_TAG_PREFIX}.v1.{DEMO_CUSTOM_VIEW_ID}"
DEMO_CUSTOM_VIEW_NAME = "Span review"
DEMO_CUSTOM_VIEW_LABEL = "Span inputs, outputs, and accuracy"
DEMO_CUSTOM_VIEW_INSTRUCTION = (
    "Show each span's input and output as cards and collect a per-span Accuracy "
    "rating from Super accurate to Not accurate, submitted together."
)
DEMO_CUSTOM_VIEW_CREATED_AT_MS = 1

_ACCURACY_OPTIONS = [
    {"label": "Super accurate", "value": "Super accurate"},
    {"label": "Accurate", "value": "Accurate"},
    {"label": "Somewhat accurate", "value": "Somewhat accurate"},
    {"label": "Not very accurate", "value": "Not very accurate"},
    {"label": "Not accurate", "value": "Not accurate"},
]

# Role selectors covering the demo traces. A2UI templates cannot loop over
# spans, so each card is authored up front and hidden via renderIfSpan when
# that role is absent. CHAIN nth=0 is usually the root (RAG / prompt), so
# only nth=1 is included to avoid a duplicate of the root card.
_CHILD_SPAN_CARDS: list[tuple[str, dict[str, object]]] = [
    ("embed", {"type": "EMBEDDING", "nth": 0}),
    ("retrieve", {"type": "RETRIEVER", "nth": 0}),
    ("chain1", {"type": "CHAIN", "nth": 1}),
    ("llm0", {"type": "LLM", "nth": 0}),
    ("llm1", {"type": "LLM", "nth": 1}),
    ("llm2", {"type": "LLM", "nth": 2}),
    ("tool0", {"type": "TOOL", "nth": 0}),
    ("tool1", {"type": "TOOL", "nth": 1}),
]


def _span_field(span_ref: str | dict[str, object], field: str) -> dict[str, object]:
    return {"$source": "spanField", "spanRef": span_ref, "field": field}


def _span_ref_marker(span_ref: str | dict[str, object]) -> dict[str, object]:
    return {"$spanRef": span_ref}


def _span_io_card(
    prefix: str, span_ref: str | dict[str, object]
) -> tuple[str, list[dict[str, object]]]:
    card_id = f"{prefix}-card"
    col_id = f"{prefix}-col"
    title_id = f"{prefix}-title"
    in_id = f"{prefix}-in"
    out_id = f"{prefix}-out"
    accuracy_id = f"{prefix}-accuracy"
    why_id = f"{prefix}-why"
    span_id = _span_ref_marker(span_ref)
    return card_id, [
        {"id": card_id, "component": "Card", "child": col_id, "renderIfSpan": span_ref},
        {
            "id": col_id,
            "component": "Column",
            "children": [title_id, in_id, out_id, accuracy_id, why_id],
        },
        {
            "id": title_id,
            "component": "Text",
            "variant": "h4",
            "text": _span_field(span_ref, "name"),
        },
        {
            "id": in_id,
            "component": "KeyValueViewer",
            "label": "Input",
            "value": _span_field(span_ref, "inputs"),
            "initialFormat": "json",
        },
        {
            "id": out_id,
            "component": "KeyValueViewer",
            "label": "Output",
            "value": _span_field(span_ref, "outputs"),
            "initialFormat": "json",
        },
        {
            "id": accuracy_id,
            "component": "RadioGroup",
            "label": "Accuracy",
            "name": "Accuracy",
            "formId": "feedback",
            "spanId": span_id,
            "options": _ACCURACY_OPTIONS,
        },
        {
            "id": why_id,
            "component": "FeedbackInputText",
            "label": "Why?",
            "name": "Accuracy",
            "field": "rationale",
            "formId": "feedback",
            "spanId": span_id,
            "placeholder": "Optional rationale",
        },
    ]


def build_demo_custom_view() -> dict[str, object]:
    """Return the stored CustomView payload for the seeded demo view."""
    root_card_id, root_components = _span_io_card("root", "root")
    child_card_ids: list[str] = []
    child_components: list[dict[str, object]] = []
    for prefix, selector in _CHILD_SPAN_CARDS:
        card_id, components = _span_io_card(prefix, selector)
        child_card_ids.append(card_id)
        child_components.extend(components)

    components = [
        {
            "id": "root",
            "component": "Column",
            "children": ["metrics", root_card_id, *child_card_ids, "submit"],
        },
        {
            "id": "metrics",
            "component": "Row",
            "align": "stretch",
            "children": ["stat-status", "stat-latency", "stat-tokens"],
        },
        {
            "id": "stat-status",
            "component": "StatCard",
            "value": {"$source": "metrics.status"},
            "label": "Status",
            "icon": "checklist",
            "tone": "info",
        },
        {
            "id": "stat-latency",
            "component": "StatCard",
            "value": {"$source": "metrics.latency"},
            "label": "Latency",
            "icon": "clock",
            "tone": "info",
        },
        {
            "id": "stat-tokens",
            "component": "StatCard",
            "value": {"$source": "metrics.totalTokens"},
            "label": "Tokens",
            "icon": "hash",
            "tone": "info",
        },
        *root_components,
        *child_components,
        {
            "id": "submit",
            "component": "FeedbackSubmit",
            "label": "Submit feedback",
            "formId": "feedback",
        },
    ]
    return {
        "id": DEMO_CUSTOM_VIEW_ID,
        "name": DEMO_CUSTOM_VIEW_NAME,
        "label": DEMO_CUSTOM_VIEW_LABEL,
        "instruction": DEMO_CUSTOM_VIEW_INSTRUCTION,
        "template": [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "main",
                    "components": components,
                },
            }
        ],
        "createdAtMs": DEMO_CUSTOM_VIEW_CREATED_AT_MS,
    }


def serialize_demo_custom_view() -> str:
    payload = json.dumps(build_demo_custom_view(), separators=(",", ":"))
    if len(payload) > MAX_EXPERIMENT_TAG_VAL_LENGTH:
        raise ValueError(
            f"Demo custom view exceeds experiment tag limit "
            f"({len(payload)} > {MAX_EXPERIMENT_TAG_VAL_LENGTH})"
        )
    return payload


class CustomViewDemoGenerator(BaseDemoGenerator):
    """Seeds a saved custom view on the demo experiment.

    Writes one experiment tag (`mlflow.customView.view.v1.<id>`) whose template
    binds per-span input/output cards and an Accuracy feedback form. The host
    re-resolves the bindings for whichever demo trace is open.
    """

    name = DemoFeature.CUSTOM_VIEW
    version = 1

    def generate(self) -> DemoResult:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found")

        experiment_id = experiment.experiment_id
        MlflowClient().set_experiment_tag(
            experiment_id, DEMO_CUSTOM_VIEW_TAG_KEY, serialize_demo_custom_view()
        )
        return DemoResult(
            feature=self.name,
            entity_ids=[DEMO_CUSTOM_VIEW_ID],
            navigation_url=f"#/experiments/{experiment_id}",
        )

    def _data_exists(self) -> bool:
        try:
            experiment = _get_store().get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None or experiment.lifecycle_stage != "active":
                return False
            return bool(experiment.tags.get(DEMO_CUSTOM_VIEW_TAG_KEY))
        except Exception:
            _logger.warning("Failed to check if custom view demo exists", exc_info=True)
            return False

    def delete_demo(self) -> None:
        try:
            experiment = _get_store().get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None:
                return
            MlflowClient().delete_experiment_tag(experiment.experiment_id, DEMO_CUSTOM_VIEW_TAG_KEY)
        except Exception:
            _logger.warning("Failed to delete demo custom view", exc_info=True)
