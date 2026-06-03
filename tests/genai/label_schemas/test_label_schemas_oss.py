"""Tests for the label-schema fluent SDK on a (non-Databricks) tracking store.

The same functions route to the Databricks ReviewApp on a Databricks tracking
URI; that path is exercised in ``test_label_schemas.py`` (which skips when
``databricks-agents`` is unavailable). Here we force the tracking-store path
and assert delegation to ``TracingClient``.
"""

from unittest.mock import patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas import (
    InputCategorical,
    InputPassFail,
    LabelSchema,
    LabelSchemaType,
    create_label_schema,
    delete_label_schema,
    get_label_schema,
    list_label_schemas,
    update_label_schema,
)
from mlflow.store.entities.paged_list import PagedList

_BASE = "mlflow.genai.label_schemas.TracingClient"


@pytest.fixture(autouse=True)
def _force_tracking_store_path():
    # Route the dispatching fluent functions to the tracking-store path
    # regardless of the ambient tracking URI.
    with patch("mlflow.genai.label_schemas.is_databricks_uri", return_value=False):
        yield


def _pass_fail_schema(schema_id="ls-1"):
    return LabelSchema(
        schema_id=schema_id,
        experiment_id="1",
        name="Is the answer correct?",
        type=LabelSchemaType.FEEDBACK,
        input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
    )


def test_create_label_schema_delegates_to_tracing_client():
    with patch(f"{_BASE}._create_label_schema", return_value=_pass_fail_schema()) as mock_create:
        result = create_label_schema(
            name="Is the answer correct?",
            type="feedback",
            input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
            instruction="Mark Correct if accurate.",
            enable_comment=True,
            experiment_id="1",
        )
        assert result.schema_id == "ls-1"
        kwargs = mock_create.call_args[1]
        assert kwargs["experiment_id"] == "1"
        assert kwargs["name"] == "Is the answer correct?"
        assert kwargs["type"] == "feedback"
        assert kwargs["enable_comment"] is True
        assert isinstance(kwargs["input"], InputPassFail)


def test_create_label_schema_defaults_experiment_id_to_current():
    with (
        patch(f"{_BASE}._create_label_schema", return_value=_pass_fail_schema()) as mock_create,
        patch("mlflow.tracking.fluent._get_experiment_id", return_value="7"),
    ):
        create_label_schema(
            name="x",
            type="feedback",
            input=InputPassFail(positive_label="a", negative_label="b"),
        )
        assert mock_create.call_args[1]["experiment_id"] == "7"


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"title": "T"}, "title"), ({"overwrite": True}, "overwrite")]
)
def test_create_label_schema_rejects_databricks_only_params(kwargs, match):
    with pytest.raises(MlflowException, match=match):
        create_label_schema(
            name="x",
            type="feedback",
            input=InputPassFail(positive_label="a", negative_label="b"),
            experiment_id="1",
            **kwargs,
        )


def test_get_label_schema_by_id_delegates():
    with patch(f"{_BASE}._get_label_schema", return_value=_pass_fail_schema()) as mock_get:
        result = get_label_schema(schema_id="ls-1")
        assert result.schema_id == "ls-1"
        mock_get.assert_called_once_with("ls-1")


def test_get_label_schema_by_name_delegates():
    with patch(f"{_BASE}._get_label_schema_by_name", return_value=_pass_fail_schema()) as mock_get:
        result = get_label_schema(name="Is the answer correct?", experiment_id="1")
        assert result.name == "Is the answer correct?"
        mock_get.assert_called_once_with("1", "Is the answer correct?")


def test_get_label_schema_requires_id_or_experiment_and_name():
    with pytest.raises(MlflowException, match="Provide `schema_id`"):
        get_label_schema(name="x")


def test_list_label_schemas_paginates():
    paged = PagedList([_pass_fail_schema("ls-1"), _pass_fail_schema("ls-2")], "tok")
    with patch(f"{_BASE}._list_label_schemas", return_value=paged) as mock_list:
        result = list_label_schemas("1", max_results=2, page_token="prev")
        assert len(result) == 2
        assert result.token == "tok"
        mock_list.assert_called_once_with("1", max_results=2, page_token="prev")


def test_update_label_schema_sparse_kwargs():
    with patch(f"{_BASE}._update_label_schema", return_value=_pass_fail_schema()) as mock_update:
        update_label_schema("ls-1", instruction="Updated instruction")
        kwargs = mock_update.call_args[1]
        # name/enable_comment/input default to None and are passed through; the
        # RestStore is responsible for HasField-driven serialization.
        assert kwargs["instruction"] == "Updated instruction"
        assert kwargs["name"] is None
        assert kwargs["enable_comment"] is None
        assert kwargs["input"] is None


def test_update_label_schema_replaces_input():
    with patch(f"{_BASE}._update_label_schema", return_value=_pass_fail_schema()) as mock_update:
        new_input = InputCategorical(options=["low", "high"])
        update_label_schema("ls-1", input=new_input)
        assert mock_update.call_args[1]["input"] is new_input


def test_delete_label_schema_by_id_delegates():
    with patch(f"{_BASE}._delete_label_schema") as mock_delete:
        delete_label_schema(schema_id="ls-1")
        mock_delete.assert_called_once_with("ls-1")


@pytest.mark.parametrize(
    "call",
    [
        lambda: list_label_schemas("1"),
        lambda: update_label_schema("ls-1", instruction="x"),
    ],
    ids=["list", "update"],
)
def test_list_and_update_unsupported_on_databricks(call):
    with (
        patch("mlflow.genai.label_schemas.is_databricks_uri", return_value=True),
        pytest.raises(MlflowException, match="not supported on a Databricks"),
    ):
        call()
