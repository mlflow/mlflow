from __future__ import annotations

import base64
import json
import logging
import time
import zlib
from dataclasses import dataclass
from typing import Any

from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.entities import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_EXPERIMENT_TAG_VAL_LENGTH

_logger = logging.getLogger(__name__)

TRACE_V4_SAVED_VIEW_TAG_PREFIX = "mlflow.tracesV4ViewState."
_DEFLATE_PREFIX = "deflate;"

DEMO_TRACE_V4_MLFLOW_VIEW_ID = "mlflow-demo-traces-mlflow"


@dataclass(frozen=True)
class DemoTraceSavedViewDef:
    id: str
    name: str
    tag_prefix: str
    state: dict[str, Any]

    @property
    def tag_key(self) -> str:
        return f"{self.tag_prefix}{self.id}"


DEMO_TRACE_V4_SAVED_VIEW = DemoTraceSavedViewDef(
    id=DEMO_TRACE_V4_MLFLOW_VIEW_ID,
    name="MLflow conversations",
    tag_prefix=TRACE_V4_SAVED_VIEW_TAG_PREFIX,
    state={
        "single": {
            "q": "MLflow",
            "pageSize": "50",
            "startTimeLabel": "ALL",
            "cols": "start_time,session,input,output,duration,state,tokens",
        },
        "multi": {},
    },
)

DEMO_TRACE_SAVED_VIEWS = [
    DEMO_TRACE_V4_SAVED_VIEW,
]

DEMO_TRACE_SAVED_VIEW_TAG_KEYS = [view.tag_key for view in DEMO_TRACE_SAVED_VIEWS]


def _compress_state(state: dict[str, Any]) -> str:
    serialized = json.dumps(state, separators=(",", ":"))
    compressed = base64.b64encode(zlib.compress(serialized.encode("utf-8"))).decode("ascii")
    return f"{_DEFLATE_PREFIX}{compressed}"


def _serialize_trace_saved_view(view: DemoTraceSavedViewDef, created_at_ms: int) -> str:
    payload = json.dumps(
        {
            "name": view.name,
            "createdAt": created_at_ms,
            "state": _compress_state(view.state),
        },
        separators=(",", ":"),
    )
    if len(payload) > MAX_EXPERIMENT_TAG_VAL_LENGTH:
        raise ValueError(
            f"Demo saved view '{view.id}' exceeds experiment tag limit "
            f"({len(payload)} > {MAX_EXPERIMENT_TAG_VAL_LENGTH})"
        )
    return payload


def _is_resource_missing(exc: MlflowException) -> bool:
    return exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


class SavedViewsDemoGenerator(BaseDemoGenerator):
    """Seeds saved V4 Traces views on the demo experiment."""

    name = DemoFeature.SAVED_VIEWS
    version = 1

    def generate(self) -> DemoResult:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found")

        experiment_id = experiment.experiment_id
        client = MlflowClient()

        created_at_ms = int(time.time() * 1000)
        for view in DEMO_TRACE_SAVED_VIEWS:
            client.set_experiment_tag(
                experiment_id,
                view.tag_key,
                _serialize_trace_saved_view(view, created_at_ms),
            )

        return DemoResult(
            feature=self.name,
            entity_ids=DEMO_TRACE_SAVED_VIEW_TAG_KEYS,
            navigation_url=f"#/experiments/{experiment_id}/traces",
        )

    def _data_exists(self) -> bool:
        try:
            experiment = _get_store().get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None or experiment.lifecycle_stage != LifecycleStage.ACTIVE:
                return False
            return all(tag_key in experiment.tags for tag_key in DEMO_TRACE_SAVED_VIEW_TAG_KEYS)
        except MlflowException as e:
            if not _is_resource_missing(e):
                raise
            _logger.debug("Failed to check if saved views demo exists", exc_info=True)
            return False

    def delete_demo(self) -> None:
        experiment = _get_store().get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            return
        client = MlflowClient()
        for tag_key in DEMO_TRACE_SAVED_VIEW_TAG_KEYS:
            try:
                client.delete_experiment_tag(experiment.experiment_id, tag_key)
            except MlflowException as e:
                if not _is_resource_missing(e):
                    raise
