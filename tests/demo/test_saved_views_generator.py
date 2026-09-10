import base64
import json
import time
import zlib
from unittest import mock

import pytest

import mlflow
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.saved_views import (
    DEMO_TRACE_SAVED_VIEW_TAG_KEYS,
    DEMO_TRACE_SAVED_VIEWS,
    DEMO_TRACE_V4_MLFLOW_VIEW_ID,
    DEMO_TRACE_V4_SAVED_VIEW,
    SavedViewsDemoGenerator,
    _serialize_trace_saved_view,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.validation import MAX_EXPERIMENT_TAG_VAL_LENGTH


def _create_demo_experiment():
    return mlflow.set_experiment(DEMO_EXPERIMENT_NAME)


def _inflate_envelope(value: str) -> dict[str, object]:
    envelope = json.loads(value)
    assert envelope["state"].startswith("deflate;"), (
        f"expected deflate-compressed state, got: {envelope['state'][:40]!r}"
    )
    compressed = base64.b64decode(envelope["state"].removeprefix("deflate;"))
    return json.loads(zlib.decompress(compressed).decode("utf-8"))


def test_generator_attributes():
    generator = SavedViewsDemoGenerator()
    assert generator.name == DemoFeature.SAVED_VIEWS
    assert generator.version == 1
    assert {view.id for view in DEMO_TRACE_SAVED_VIEWS} == {DEMO_TRACE_V4_MLFLOW_VIEW_ID}
    assert DEMO_TRACE_SAVED_VIEW_TAG_KEYS == [DEMO_TRACE_V4_SAVED_VIEW.tag_key]
    assert DEMO_TRACE_V4_SAVED_VIEW.tag_key.startswith("mlflow.tracesV4ViewState.")


@pytest.mark.parametrize("view", DEMO_TRACE_SAVED_VIEWS)
def test_serialized_view_fits_experiment_tag_limit(view):
    payload = _serialize_trace_saved_view(view, created_at_ms=123)
    assert len(payload) <= MAX_EXPERIMENT_TAG_VAL_LENGTH


@pytest.mark.parametrize("view", DEMO_TRACE_SAVED_VIEWS)
def test_serialized_view_shape(view):
    payload = _serialize_trace_saved_view(view, created_at_ms=123)
    envelope = json.loads(payload)
    assert envelope["name"] == view.name
    assert envelope["createdAt"] == 123

    state = _inflate_envelope(payload)
    assert state == view.state
    assert "single" in state
    assert "multi" in state
    assert state["single"]["startTimeLabel"] == "ALL"


def test_v4_serialized_view_uses_v4_state_keys():
    view = DEMO_TRACE_V4_SAVED_VIEW
    state = _inflate_envelope(_serialize_trace_saved_view(view, created_at_ms=123))
    assert "q" in state["single"]
    assert "cols" in state["single"]


def test_data_exists_false_when_no_experiment():
    generator = SavedViewsDemoGenerator()
    assert generator._data_exists() is False


def test_data_exists_raises_unexpected_store_failure():
    generator = SavedViewsDemoGenerator()
    store = mock.MagicMock()
    store.get_experiment_by_name.side_effect = RuntimeError("boom")

    with mock.patch("mlflow.demo.generators.saved_views._get_store", return_value=store):
        with pytest.raises(RuntimeError, match="boom"):
            generator._data_exists()


def test_data_exists_false_when_store_resource_is_missing():
    generator = SavedViewsDemoGenerator()
    store = mock.MagicMock()
    store.get_experiment_by_name.side_effect = MlflowException(
        "missing", error_code=RESOURCE_DOES_NOT_EXIST
    )

    with mock.patch("mlflow.demo.generators.saved_views._get_store", return_value=store):
        assert generator._data_exists() is False


def test_data_exists_raises_unexpected_mlflow_exception():
    generator = SavedViewsDemoGenerator()
    store = mock.MagicMock()
    store.get_experiment_by_name.side_effect = MlflowException(
        "invalid", error_code=INVALID_PARAMETER_VALUE
    )

    with mock.patch("mlflow.demo.generators.saved_views._get_store", return_value=store):
        with pytest.raises(MlflowException, match="invalid"):
            generator._data_exists()


def test_generate_requires_demo_experiment():
    generator = SavedViewsDemoGenerator()
    with pytest.raises(ValueError, match="not found"):
        generator.generate()


def test_generate_writes_experiment_tags():
    _create_demo_experiment()
    generator = SavedViewsDemoGenerator()
    before_generate_ms = int(time.time() * 1000)
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.SAVED_VIEWS
    assert result.entity_ids == DEMO_TRACE_SAVED_VIEW_TAG_KEYS
    assert "/experiments/" in result.navigation_url
    assert result.navigation_url.endswith("/traces")

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    for view in DEMO_TRACE_SAVED_VIEWS:
        assert view.tag_key in experiment.tags
        envelope = json.loads(experiment.tags[view.tag_key])
        assert before_generate_ms <= envelope["createdAt"] <= int(time.time() * 1000)
        assert _inflate_envelope(experiment.tags[view.tag_key]) == view.state


def test_data_exists_requires_all_tags():
    _create_demo_experiment()
    generator = SavedViewsDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()
    assert generator._data_exists() is True

    mlflow.MlflowClient().delete_experiment_tag(
        mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME).experiment_id,
        DEMO_TRACE_SAVED_VIEW_TAG_KEYS[0],
    )
    assert generator._data_exists() is False


def test_delete_demo_removes_tags():
    _create_demo_experiment()
    generator = SavedViewsDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert all(key not in experiment.tags for key in DEMO_TRACE_SAVED_VIEW_TAG_KEYS)


def test_delete_demo_raises_unexpected_tag_delete_failure():
    _create_demo_experiment()
    generator = SavedViewsDemoGenerator()
    generator.generate()

    with mock.patch("mlflow.tracking.client.MlflowClient.delete_experiment_tag") as delete_tag:
        delete_tag.side_effect = MlflowException(
            "cannot delete", error_code=INVALID_PARAMETER_VALUE
        )

        with pytest.raises(MlflowException, match="cannot delete"):
            generator.delete_demo()


def test_is_generated_checks_version(monkeypatch):
    _create_demo_experiment()
    saved_views_generator = SavedViewsDemoGenerator()
    saved_views_generator.generate()
    saved_views_generator.store_version()

    assert saved_views_generator.is_generated() is True

    monkeypatch.setattr(SavedViewsDemoGenerator, "version", 99)
    fresh_generator = SavedViewsDemoGenerator()
    assert fresh_generator._data_exists() is True
    assert fresh_generator.is_generated() is False
    assert fresh_generator._data_exists() is False
