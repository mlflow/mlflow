"""Unit tests for the OSS-native label-schema REST handlers.

These tests exercise the handler functions directly (mocking the tracking
store and request-message parsing); the wire-format round-trip through
``LabelSchema.to_proto`` / ``from_proto`` is covered separately.
"""

import json
from unittest import mock

import pytest

from mlflow.genai.label_schemas.label_schemas import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    LabelSchema,
    LabelSchemaType,
)
from mlflow.protos.label_schemas_pb2 import (
    FEEDBACK,
    LABEL_SCHEMA_TYPE_UNSPECIFIED,
    CreateLabelSchema,
    DeleteLabelSchema,
    GetLabelSchema,
    GetLabelSchemaByName,
    LabelSchemaInput,
    ListLabelSchemas,
    UpdateLabelSchema,
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
)
from mlflow.store.entities.paged_list import PagedList

_BASE_PATCH = "mlflow.server.handlers"


def _pass_fail_entity(schema_id: str = "ls-1", experiment_id: str = "1") -> LabelSchema:
    return LabelSchema(
        schema_id=schema_id,
        experiment_id=experiment_id,
        name="correctness",
        type=LabelSchemaType.FEEDBACK,
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


def test_create_label_schema_routes_categorical_input():
    request_message = CreateLabelSchema(
        experiment_id="1",
        name="severity",
        type=FEEDBACK,
        input=LabelSchemaInput(
            categorical=ProtoInputCategorical(
                options=["low", "medium", "high"],
                multi_select=True,
            )
        ),
    )
    entity = LabelSchema(
        schema_id="ls-2",
        experiment_id="1",
        name="severity",
        type=LabelSchemaType.FEEDBACK,
        input=InputCategorical(options=["low", "medium", "high"], multi_select=True),
        created_at=1000,
        updated_at=1000,
    )
    store, response = _run_handler(
        _create_label_schema, request_message, "create_label_schema", entity
    )
    call_kwargs = store.create_label_schema.call_args[1]
    assert isinstance(call_kwargs["input"], InputCategorical)
    assert call_kwargs["input"].options == ["low", "medium", "high"]
    assert call_kwargs["input"].multi_select is True

    body = json.loads(response.get_data())
    cat_body = body["label_schema"]["input"]["categorical"]
    assert cat_body["multi_select"] is True


@pytest.mark.parametrize(
    ("handler", "request_message", "store_attr"),
    [
        (
            _update_label_schema,
            UpdateLabelSchema(schema_id="ls-1", input=LabelSchemaInput()),
            "update_label_schema",
        ),
    ],
)
def test_handler_rejects_empty_oneof(handler, request_message, store_attr):
    with (
        mock.patch(f"{_BASE_PATCH}._get_tracking_store") as mock_store,
        mock.patch(f"{_BASE_PATCH}._get_request_message", return_value=request_message),
    ):
        response = handler()
        assert response.status_code == 400
        assert json.loads(response.get_data())["error_code"] == "INVALID_PARAMETER_VALUE"
        getattr(mock_store.return_value, store_attr).assert_not_called()


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
        schema_id="ls-1", name="Updated name", instruction="Updated instruction"
    )
    store, response = _run_handler(
        _update_label_schema, request_message, "update_label_schema", _pass_fail_entity()
    )
    call_kwargs = store.update_label_schema.call_args[1]
    # Exact-equality is intentional: any extra key here means HasField didn't
    # gate the sparse update correctly and would corrupt the store-layer's
    # "None=unchanged" contract.
    assert call_kwargs == {"name": "Updated name", "instruction": "Updated instruction"}
    store.update_label_schema.assert_called_once()


@pytest.mark.parametrize(
    ("enable_comment_set", "expected_in_kwargs", "expected_value"),
    [
        (True, True, True),
        (False, True, False),
        (None, False, None),  # omit -> kwargs should not contain enable_comment
    ],
    ids=["set-true", "set-false-explicit", "omit"],
)
def test_update_label_schema_enable_comment_hasfield_gate(
    enable_comment_set, expected_in_kwargs, expected_value
):
    # Stack 1's update contract: None means "unchanged"; the wire must
    # distinguish "user said False" from "user didn't say". Verified end-to-end
    # by sending enable_comment explicitly (True or False) vs omitting it.
    request_message = UpdateLabelSchema(schema_id="ls-1")
    if enable_comment_set is not None:
        request_message.enable_comment = enable_comment_set
    store, _ = _run_handler(
        _update_label_schema, request_message, "update_label_schema", _pass_fail_entity()
    )
    call_kwargs = store.update_label_schema.call_args[1]
    if expected_in_kwargs:
        assert call_kwargs["enable_comment"] is expected_value
    else:
        assert "enable_comment" not in call_kwargs


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


@pytest.mark.parametrize("bad_max_results", [0, -1, 100_000])
def test_list_label_schemas_max_results_bound_check(bad_max_results):
    # The handler's schema dict wires max_results through _assert_intlike +
    # _assert_intlike_within_range; the dispatcher inside _get_request_message
    # converts an AssertionError into MlflowException(INVALID_PARAMETER_VALUE).
    # Hit the schema directly so we don't have to spin up a Flask app.
    from mlflow.server.handlers import (
        _assert_intlike,
        _assert_intlike_within_range,
        _validate_param_against_schema,
    )
    from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD

    schema_fns = [
        _assert_intlike,
        lambda x: _assert_intlike_within_range(
            int(x),
            1,
            SEARCH_MAX_RESULTS_THRESHOLD,
            message=f"max_results must be between 1 and {SEARCH_MAX_RESULTS_THRESHOLD}.",
        ),
    ]
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match="max_results") as exc_info:
        _validate_param_against_schema(
            schema=schema_fns,
            param="max_results",
            value=bad_max_results,
            proto_parsing_succeeded=False,
        )
    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"


def test_delete_label_schema():
    request_message = DeleteLabelSchema(schema_id="ls-1")
    store, response = _run_handler(
        _delete_label_schema, request_message, "delete_label_schema", None
    )
    store.delete_label_schema.assert_called_once_with("ls-1")
    body = json.loads(response.get_data())
    assert body == {}
