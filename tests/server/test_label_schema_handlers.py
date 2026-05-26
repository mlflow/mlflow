"""Unit tests for the OSS-native label-schema REST handlers.

These tests exercise the handler functions directly (mocking the tracking
store and request-message parsing); the wire-format round-trip through
``LabelSchema.to_proto`` / ``from_proto`` is covered separately.
"""

import json
from unittest import mock

from mlflow.genai.label_schemas.label_schemas import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    LabelSchema,
    LabelSchemaType,
)
from mlflow.protos.label_schemas_pb2 import (
    ASCENDING,
    EXPECTATION,
    FEEDBACK,
    LABEL_SCHEMA_TYPE_UNSPECIFIED,
    CreateLabelSchema,
    DeleteLabelSchema,
    GetLabelSchema,
    GetLabelSchemaByName,
    LabelSchemaInput,
    ListLabelSchemas,
    UpdateLabelSchema,
    UpsertLabelSchema,
)
from mlflow.protos.label_schemas_pb2 import (
    InputCategorical as ProtoInputCategorical,
)
from mlflow.protos.label_schemas_pb2 import (
    InputNumeric as ProtoInputNumeric,
)
from mlflow.protos.label_schemas_pb2 import (
    InputPassFail as ProtoInputPassFail,
)
from mlflow.server.handlers import (
    _create_label_schema,
    _delete_label_schema,
    _get_label_schema,
    _get_label_schema_by_name,
    _list_label_schemas,
    _update_label_schema,
    _upsert_label_schema,
)
from mlflow.store.entities.paged_list import PagedList

_BASE_PATCH = "mlflow.server.handlers"


def _pass_fail_entity(schema_id: str = "ls-1", experiment_id: str = "1") -> LabelSchema:
    return LabelSchema(
        schema_id=schema_id,
        experiment_id=experiment_id,
        name="correctness",
        type=LabelSchemaType.FEEDBACK,
        title="Is the answer correct?",
        input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
        instruction="Mark Correct if accurate.",
        enable_comment=True,
        created_by="alice@example.com",
        created_at=1000,
        updated_at=1000,
    )


def _run_handler(handler, request_message, store_attr: str, return_value):
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        getattr(mock_store.return_value, store_attr).return_value = return_value
        response = handler()
        return mock_store.return_value, response


def test_create_label_schema_routes_pass_fail_input():
    request_message = CreateLabelSchema(
        experiment_id="1",
        name="correctness",
        type=FEEDBACK,
        title="Is the answer correct?",
        input=LabelSchemaInput(
            pass_fail=ProtoInputPassFail(positive_label="Correct", negative_label="Incorrect")
        ),
        instruction="Mark Correct if accurate.",
        enable_comment=True,
    )
    store, response = _run_handler(
        _create_label_schema, request_message, "create_label_schema", _pass_fail_entity()
    )
    call_kwargs = store.create_label_schema.call_args[1]
    assert call_kwargs["experiment_id"] == "1"
    assert call_kwargs["name"] == "correctness"
    assert call_kwargs["type"] == LabelSchemaType.FEEDBACK
    assert call_kwargs["title"] == "Is the answer correct?"
    assert isinstance(call_kwargs["input"], InputPassFail)
    assert call_kwargs["input"].positive_label == "Correct"
    assert call_kwargs["instruction"] == "Mark Correct if accurate."
    assert call_kwargs["enable_comment"] is True

    body = json.loads(response.get_data())
    assert body["label_schema"]["schema_id"] == "ls-1"
    assert body["label_schema"]["input"]["pass_fail"]["positive_label"] == "Correct"


def test_create_label_schema_rejects_unspecified_type():
    request_message = CreateLabelSchema(
        experiment_id="1",
        name="x",
        type=LABEL_SCHEMA_TYPE_UNSPECIFIED,
        title="X",
        input=LabelSchemaInput(
            pass_fail=ProtoInputPassFail(positive_label="y", negative_label="n")
        ),
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _create_label_schema()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.create_label_schema.assert_not_called()


def test_create_label_schema_rejects_empty_oneof():
    request_message = CreateLabelSchema(
        experiment_id="1",
        name="x",
        type=FEEDBACK,
        title="X",
        input=LabelSchemaInput(),
    )
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = _create_label_schema()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        mock_store.return_value.create_label_schema.assert_not_called()


def test_create_label_schema_categorical_round_trip():
    request_message = CreateLabelSchema(
        experiment_id="1",
        name="severity",
        type=FEEDBACK,
        title="Severity",
        input=LabelSchemaInput(
            categorical=ProtoInputCategorical(
                options=["low", "medium", "high"],
                semantic_polarity=ASCENDING,
                multi_select=True,
            )
        ),
    )
    entity = LabelSchema(
        schema_id="ls-2",
        experiment_id="1",
        name="severity",
        type=LabelSchemaType.FEEDBACK,
        title="Severity",
        input=InputCategorical(
            options=["low", "medium", "high"], semantic_polarity="ascending", multi_select=True
        ),
        created_at=1000,
        updated_at=1000,
    )
    store, response = _run_handler(
        _create_label_schema, request_message, "create_label_schema", entity
    )
    call_kwargs = store.create_label_schema.call_args[1]
    assert isinstance(call_kwargs["input"], InputCategorical)
    assert call_kwargs["input"].options == ["low", "medium", "high"]
    assert call_kwargs["input"].semantic_polarity == "ascending"
    assert call_kwargs["input"].multi_select is True

    body = json.loads(response.get_data())
    assert body["label_schema"]["input"]["categorical"]["multi_select"] is True


def test_get_label_schema():
    request_message = GetLabelSchema(schema_id="ls-1")
    store, response = _run_handler(
        _get_label_schema, request_message, "get_label_schema", _pass_fail_entity()
    )
    store.get_label_schema.assert_called_once_with("ls-1")
    body = json.loads(response.get_data())
    assert body["label_schema"]["schema_id"] == "ls-1"


def test_get_label_schema_by_name():
    request_message = GetLabelSchemaByName(experiment_id="1", name="correctness")
    store, response = _run_handler(
        _get_label_schema_by_name,
        request_message,
        "get_label_schema_by_name",
        _pass_fail_entity(),
    )
    store.get_label_schema_by_name.assert_called_once_with("1", "correctness")
    body = json.loads(response.get_data())
    assert body["label_schema"]["name"] == "correctness"


def test_list_label_schemas_default_max_results():
    request_message = ListLabelSchemas(experiment_id="1")
    paged = PagedList([_pass_fail_entity("ls-1"), _pass_fail_entity("ls-2")], "tok")
    store, response = _run_handler(
        _list_label_schemas, request_message, "list_label_schemas", paged
    )
    store.list_label_schemas.assert_called_once_with("1", max_results=100, page_token=None)
    body = json.loads(response.get_data())
    assert len(body["label_schemas"]) == 2
    assert body["next_page_token"] == "tok"


def test_list_label_schemas_with_pagination_args():
    request_message = ListLabelSchemas(experiment_id="1", max_results=2, page_token="pt")
    paged = PagedList([_pass_fail_entity("ls-3")], None)
    store, response = _run_handler(
        _list_label_schemas, request_message, "list_label_schemas", paged
    )
    store.list_label_schemas.assert_called_once_with("1", max_results=2, page_token="pt")
    body = json.loads(response.get_data())
    assert len(body["label_schemas"]) == 1
    assert "next_page_token" not in body or body.get("next_page_token", "") == ""


def test_update_label_schema_sparse_fields():
    request_message = UpdateLabelSchema(
        schema_id="ls-1", title="Updated title", instruction="Updated instruction"
    )
    store, response = _run_handler(
        _update_label_schema, request_message, "update_label_schema", _pass_fail_entity()
    )
    call_kwargs = store.update_label_schema.call_args[1]
    assert call_kwargs == {"title": "Updated title", "instruction": "Updated instruction"}


def test_update_label_schema_input_replace():
    request_message = UpdateLabelSchema(
        schema_id="ls-1",
        input=LabelSchemaInput(numeric=ProtoInputNumeric(min_value=1.0, max_value=5.0)),
    )
    store, _ = _run_handler(
        _update_label_schema, request_message, "update_label_schema", _pass_fail_entity()
    )
    call_kwargs = store.update_label_schema.call_args[1]
    assert isinstance(call_kwargs["input"], InputNumeric)
    assert call_kwargs["input"].min_value == 1.0
    assert call_kwargs["input"].max_value == 5.0


def test_upsert_label_schema():
    request_message = UpsertLabelSchema(
        experiment_id="1",
        name="rating",
        type=EXPECTATION,
        title="Rating",
        input=LabelSchemaInput(numeric=ProtoInputNumeric(min_value=1.0, max_value=5.0)),
    )
    store, response = _run_handler(
        _upsert_label_schema, request_message, "upsert_label_schema", _pass_fail_entity()
    )
    call_kwargs = store.upsert_label_schema.call_args[1]
    assert call_kwargs["experiment_id"] == "1"
    assert call_kwargs["type"] == LabelSchemaType.EXPECTATION
    # enable_comment + instruction omitted on the wire -> not in kwargs
    assert "enable_comment" not in call_kwargs
    assert "instruction" not in call_kwargs


def test_delete_label_schema():
    request_message = DeleteLabelSchema(schema_id="ls-1")
    store, response = _run_handler(
        _delete_label_schema, request_message, "delete_label_schema", None
    )
    store.delete_label_schema.assert_called_once_with("ls-1")
    body = json.loads(response.get_data())
    assert body == {}
