"""Tests for the OSS-native label-schema fluent SDK.

The Databricks-routed fluent functions (``create_label_schema``,
``get_label_schema``, ``delete_label_schema``) are exercised in
``test_label_schemas.py`` which skips when ``databricks-agents`` is
unavailable. This module covers the new OSS-native experiment-scoped
fluent functions, which delegate to the tracking server's REST surface
via ``TracingClient``.
"""

from unittest.mock import patch

from mlflow.genai.label_schemas import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    LabelSchema,
    LabelSchemaType,
    create_experiment_label_schema,
    delete_experiment_label_schema,
    get_experiment_label_schema,
    get_experiment_label_schema_by_name,
    list_experiment_label_schemas,
    update_experiment_label_schema,
    upsert_experiment_label_schema,
)
from mlflow.store.entities.paged_list import PagedList

_BASE = "mlflow.genai.label_schemas.TracingClient"


def _pass_fail_schema(schema_id="ls-1"):
    return LabelSchema(
        schema_id=schema_id,
        experiment_id="1",
        name="Is the answer correct?",
        type=LabelSchemaType.FEEDBACK,
        input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
    )


def test_create_experiment_label_schema_delegates_to_tracing_client():
    with patch(f"{_BASE}._create_label_schema") as mock_create:
        mock_create.return_value = _pass_fail_schema()
        result = create_experiment_label_schema(
            experiment_id="1",
            name="Is the answer correct?",
            type="feedback",
            input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
            instruction="Mark Correct if accurate.",
            enable_comment=True,
        )
        assert result.schema_id == "ls-1"
        kwargs = mock_create.call_args[1]
        assert kwargs["experiment_id"] == "1"
        assert kwargs["name"] == "Is the answer correct?"
        assert kwargs["type"] == "feedback"
        assert kwargs["enable_comment"] is True
        assert isinstance(kwargs["input"], InputPassFail)


def test_get_experiment_label_schema_delegates():
    with patch(f"{_BASE}._get_label_schema") as mock_get:
        mock_get.return_value = _pass_fail_schema()
        result = get_experiment_label_schema("ls-1")
        assert result.schema_id == "ls-1"
        mock_get.assert_called_once_with("ls-1")


def test_get_experiment_label_schema_by_name_delegates():
    with patch(f"{_BASE}._get_label_schema_by_name") as mock_get:
        mock_get.return_value = _pass_fail_schema()
        result = get_experiment_label_schema_by_name("1", "Is the answer correct?")
        assert result.name == "Is the answer correct?"
        mock_get.assert_called_once_with("1", "Is the answer correct?")


def test_list_experiment_label_schemas_paginates():
    paged = PagedList([_pass_fail_schema("ls-1"), _pass_fail_schema("ls-2")], "tok")
    with patch(f"{_BASE}._list_label_schemas", return_value=paged) as mock_list:
        result = list_experiment_label_schemas("1", max_results=2, page_token="prev")
        assert len(result) == 2
        assert result.token == "tok"
        mock_list.assert_called_once_with("1", max_results=2, page_token="prev")


def test_update_experiment_label_schema_sparse_kwargs():
    with patch(f"{_BASE}._update_label_schema") as mock_update:
        mock_update.return_value = _pass_fail_schema()
        update_experiment_label_schema("ls-1", instruction="Updated instruction")
        kwargs = mock_update.call_args[1]
        # name/enable_comment/input default to None and are passed through;
        # the RestStore is responsible for HasField-driven serialization.
        assert kwargs["instruction"] == "Updated instruction"
        assert kwargs["name"] is None
        assert kwargs["enable_comment"] is None
        assert kwargs["input"] is None


def test_update_experiment_label_schema_replaces_input():
    with patch(f"{_BASE}._update_label_schema") as mock_update:
        mock_update.return_value = _pass_fail_schema()
        new_input = InputCategorical(options=["low", "high"])
        update_experiment_label_schema("ls-1", input=new_input)
        assert mock_update.call_args[1]["input"] is new_input


def test_upsert_experiment_label_schema_delegates():
    with patch(f"{_BASE}._upsert_label_schema") as mock_upsert:
        mock_upsert.return_value = _pass_fail_schema()
        upsert_experiment_label_schema(
            experiment_id="1",
            name="rating",
            type="expectation",
            input=InputNumeric(min_value=1.0, max_value=5.0),
        )
        kwargs = mock_upsert.call_args[1]
        assert kwargs["experiment_id"] == "1"
        assert kwargs["name"] == "rating"
        assert kwargs["type"] == "expectation"
        assert isinstance(kwargs["input"], InputNumeric)


def test_delete_experiment_label_schema_delegates():
    with patch(f"{_BASE}._delete_label_schema") as mock_delete:
        delete_experiment_label_schema("ls-1")
        mock_delete.assert_called_once_with("ls-1")
