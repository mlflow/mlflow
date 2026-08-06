import json
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from mlflow.genai.scorers import RequiredResource, scorer


def test_resource_construction():
    r = RequiredResource(type="gateway_endpoint", name="my-ep")
    assert r.type == "gateway_endpoint"
    assert r.name == "my-ep"


def test_resource_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown required resource type"):
        RequiredResource(type="unknown", name="x")


def test_resource_empty_name_raises():
    with pytest.raises(ValueError, match="name must not be empty"):
        RequiredResource(type="gateway_endpoint", name="")


def test_resource_immutable():
    r = RequiredResource(type="gateway_endpoint", name="ep")
    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        r.name = "other"


def test_resource_to_dict():
    assert RequiredResource(type="gateway_endpoint", name="ep-1").to_dict() == {
        "type": "gateway_endpoint",
        "name": "ep-1",
    }
    assert RequiredResource(type="prompt", name="prompts:/grading/1").to_dict() == {
        "type": "prompt",
        "name": "prompts:/grading/1",
    }


def test_resource_round_trip():
    original = RequiredResource(type="gateway_endpoint", name="my-ep")
    d = original.to_dict()
    restored = RequiredResource.from_dict(d)
    assert restored == original


def test_resource_from_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown required resource type"):
        RequiredResource.from_dict({"type": "unknown", "name": "x"})


def test_resource_equality():
    # Round-trip tests assert restored == original, so equality must work correctly
    assert RequiredResource(type="gateway_endpoint", name="a") == RequiredResource(
        type="gateway_endpoint", name="a"
    )
    assert RequiredResource(type="gateway_endpoint", name="a") != RequiredResource(
        type="gateway_endpoint", name="b"
    )
    assert RequiredResource(type="gateway_endpoint", name="a") != RequiredResource(
        type="prompt", name="a"
    )


def test_resource_hashable():
    # Within a Job, Downstream permission extraction unions resources across multiple
    # scorers of an experiment.
    # using sets — hashability makes deduplication automatic
    resources = {
        RequiredResource(type="gateway_endpoint", name="a"),
        RequiredResource(type="gateway_endpoint", name="a"),
        RequiredResource(type="prompt", name="b"),
    }
    assert len(resources) == 2


def test_scorer_rejects_non_required_resource():
    with pytest.raises(ValueError, match="Expected RequiredResource"):

        @scorer(required_resources=({"type": "gateway_endpoint", "name": "ep"},))
        def s(outputs):
            return True


def test_decorator_scorer_serialization_with_resources():
    @scorer(
        required_resources=(
            RequiredResource(type="gateway_endpoint", name="endpoint-1"),
            RequiredResource(type="prompt", name="prompts:/grading/1"),
        ),
    )
    def my_scorer(outputs):
        return len(str(outputs)) > 0

    serialized = my_scorer.model_dump()
    assert serialized["required_resources"] == [
        {"type": "gateway_endpoint", "name": "endpoint-1"},
        {"type": "prompt", "name": "prompts:/grading/1"},
    ]


def test_scorer_without_resources_serializes_none():
    @scorer
    def plain(outputs):
        return True

    serialized = plain.model_dump()
    assert serialized["required_resources"] is None


def test_scorer_json_round_trip():
    @scorer(
        required_resources=(RequiredResource(type="gateway_endpoint", name="ep"),),
    )
    def s(outputs):
        return True

    json_str = json.dumps(s.model_dump())

    with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
        from mlflow.genai.scorers.base import Scorer

        restored = Scorer.model_validate(json.loads(json_str))
        assert restored.required_resources == (
            RequiredResource(type="gateway_endpoint", name="ep"),
        )


def test_scorer_json_round_trip_multiple_resources():
    @scorer(
        required_resources=(
            RequiredResource(type="gateway_endpoint", name="ep-1"),
            RequiredResource(type="gateway_endpoint", name="ep-2"),
            RequiredResource(type="prompt", name="prompts:/tool/1"),
        ),
    )
    def s(outputs):
        return True

    json_str = json.dumps(s.model_dump())

    with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
        from mlflow.genai.scorers.base import Scorer

        restored = Scorer.model_validate(json.loads(json_str))
        assert len(restored.required_resources) == 3
        assert restored.required_resources[0].type == "gateway_endpoint"
        assert restored.required_resources[2].type == "prompt"


def test_scorer_json_round_trip_no_resources():
    @scorer
    def plain(outputs):
        return True

    json_str = json.dumps(plain.model_dump())

    with patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True):
        from mlflow.genai.scorers.base import Scorer

        restored = Scorer.model_validate(json.loads(json_str))
        assert restored.required_resources is None


def test_serialized_scorer_forward_compat():
    from mlflow.genai.scorers.base import SerializedScorer

    payload = {
        "name": "test",
        "call_source": "return True",
        "call_signature": "(outputs)",
        "original_func_name": "test",
        "required_resources": [{"type": "gateway_endpoint", "name": "ep"}],
        "future_field": "should be ignored",
    }
    serialized = SerializedScorer.from_dict(payload)
    assert serialized.required_resources == [{"type": "gateway_endpoint", "name": "ep"}]


def test_resource_types_match_rbac():
    from mlflow.server.auth.permissions import (
        RESOURCE_TYPE_GATEWAY_ENDPOINT,
        RESOURCE_TYPE_PROMPT,
    )

    assert (
        RequiredResource(type="gateway_endpoint", name="x").type == RESOURCE_TYPE_GATEWAY_ENDPOINT
    )
    assert RequiredResource(type="prompt", name="x").type == RESOURCE_TYPE_PROMPT
