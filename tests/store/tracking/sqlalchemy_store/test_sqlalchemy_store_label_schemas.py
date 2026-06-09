import time

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas.label_schemas import (
    InputCategorical,
    InputNumeric,
    InputPassFail,
    InputText,
    LabelSchemaType,
)
from mlflow.store.tracking.dbmodels.models import SqlLabelSchema

from tests.store.tracking.sqlalchemy_store.conftest import _create_experiments

pytestmark = pytest.mark.notrackingurimock


def _create_pass_fail_schema(store, experiment_id, name="Is the answer correct?"):
    return store.create_label_schema(
        experiment_id=experiment_id,
        name=name,
        type="feedback",
        input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
        instruction="Mark Correct if accurate.",
        enable_comment=True,
    )


def test_create_pass_fail_schema(store):
    exp_id = _create_experiments(store, "test_create_pass_fail")

    schema = _create_pass_fail_schema(store, exp_id)

    assert schema.schema_id.startswith(SqlLabelSchema.LABEL_SCHEMA_ID_PREFIX)
    assert schema.experiment_id == exp_id
    assert schema.name == "Is the answer correct?"
    assert schema.type == LabelSchemaType.FEEDBACK
    assert schema.instruction == "Mark Correct if accurate."
    assert schema.enable_comment is True
    assert isinstance(schema.input, InputPassFail)
    assert schema.input.positive_label == "Correct"
    assert schema.input.negative_label == "Incorrect"
    assert schema.created_at > 0
    assert schema.updated_at == schema.created_at


def test_create_categorical_schema(store):
    exp_id = _create_experiments(store, "test_categorical")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Severity",
        type="feedback",
        input=InputCategorical(options=["low", "medium", "high"], multi_select=False),
    )
    assert isinstance(schema.input, InputCategorical)
    assert schema.input.options == ["low", "medium", "high"]
    assert schema.input.multi_select is False


def test_create_numeric_schema_accepts_omitted_bounds(store):
    exp_id = _create_experiments(store, "test_numeric_no_bounds")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Rating",
        type="feedback",
        input=InputNumeric(),
    )
    assert isinstance(schema.input, InputNumeric)
    assert schema.input.min_value is None
    assert schema.input.max_value is None


def test_create_text_schema(store):
    exp_id = _create_experiments(store, "test_text")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Expected answer",
        type="expectation",
        input=InputText(max_length=500),
    )
    assert isinstance(schema.input, InputText)
    assert schema.input.max_length == 500


def test_create_text_schema_accepts_omitted_max_length(store):
    exp_id = _create_experiments(store, "test_text_no_max")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Notes",
        type="expectation",
        input=InputText(),
    )
    assert isinstance(schema.input, InputText)
    assert schema.input.max_length is None


def test_create_accepts_free_text_name(store):
    # `name` is the reviewer-facing prompt and is free text (no longer
    # constrained to alphanumeric + underscore).
    exp_id = _create_experiments(store, "test_free_text_name")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Is the answer correct (per the rubric)?",
        type="feedback",
        input=InputPassFail(positive_label="a", negative_label="b"),
    )
    assert schema.name == "Is the answer correct (per the rubric)?"


def test_create_rejects_duplicate_name(store):
    exp_id = _create_experiments(store, "test_duplicate")
    _create_pass_fail_schema(store, exp_id)
    with pytest.raises(MlflowException, match="already exists"):
        _create_pass_fail_schema(store, exp_id)


def test_create_rejects_missing_experiment(store):
    with pytest.raises(MlflowException, match="Experiment with id"):
        store.create_label_schema(
            experiment_id="99999",
            name="x",
            type="feedback",
            input=InputPassFail(positive_label="a", negative_label="b"),
        )


def test_get_by_id_and_by_name(store):
    exp_id = _create_experiments(store, "test_get")
    schema = _create_pass_fail_schema(store, exp_id)

    by_id = store.get_label_schema(schema.schema_id)
    assert by_id.schema_id == schema.schema_id
    assert by_id.input == schema.input

    by_name = store.get_label_schema_by_name(exp_id, "Is the answer correct?")
    assert by_name.schema_id == schema.schema_id


def test_get_by_id_missing(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema("ls-does-not-exist")


def test_get_by_name_rejects_non_integer_experiment(store):
    # A non-integer experiment ID must raise INVALID_PARAMETER_VALUE, not a raw
    # ValueError from int(...), matching the other experiment-scoped methods.
    with pytest.raises(MlflowException, match="must be a valid integer"):
        store.get_label_schema_by_name("not-an-int", "correctness")


def test_list_orders_by_created_time_desc(store):
    exp_id = _create_experiments(store, "test_list")
    # Sleep between creates so created_time values are distinct on fast
    # backends (SQLite millisecond resolution can collide otherwise).
    s1 = _create_pass_fail_schema(store, exp_id, name="first")
    time.sleep(0.005)
    s2 = _create_pass_fail_schema(store, exp_id, name="second")
    time.sleep(0.005)
    s3 = _create_pass_fail_schema(store, exp_id, name="third")

    schemas = store.list_label_schemas(exp_id)
    assert [s.name for s in schemas] == ["third", "second", "first"]
    assert [s.created_at for s in schemas] == sorted(
        [s1.created_at, s2.created_at, s3.created_at], reverse=True
    )


@pytest.mark.parametrize("bad_max_results", [None, 0, -1, 1_000_000])
def test_list_rejects_invalid_max_results(store, bad_max_results):
    exp_id = _create_experiments(store, f"test_list_validate_{bad_max_results}")
    with pytest.raises(MlflowException, match="max_results"):
        store.list_label_schemas(exp_id, max_results=bad_max_results)


def test_list_pagination(store):
    exp_id = _create_experiments(store, "test_pagination")
    for i in range(5):
        _create_pass_fail_schema(store, exp_id, name=f"schema_{i}")

    page1 = store.list_label_schemas(exp_id, max_results=2)
    assert len(page1) == 2
    assert page1.token is not None

    page2 = store.list_label_schemas(exp_id, max_results=2, page_token=page1.token)
    assert len(page2) == 2
    assert page2.token is not None

    page3 = store.list_label_schemas(exp_id, max_results=2, page_token=page2.token)
    assert len(page3) == 1
    assert page3.token is None


def test_update_rename(store):
    exp_id = _create_experiments(store, "test_rename")
    schema = _create_pass_fail_schema(store, exp_id)

    updated = store.update_label_schema(schema.schema_id, name="Answer correctness")
    assert updated.name == "Answer correctness"

    # Old name no longer findable
    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema_by_name(exp_id, "Is the answer correct?")

    # New name resolves to same schema_id
    re_fetched = store.get_label_schema_by_name(exp_id, "Answer correctness")
    assert re_fetched.schema_id == schema.schema_id


def test_update_rename_collision(store):
    exp_id = _create_experiments(store, "test_rename_collision")
    schema = _create_pass_fail_schema(store, exp_id, name="a")
    _create_pass_fail_schema(store, exp_id, name="b")

    with pytest.raises(MlflowException, match="already exists"):
        store.update_label_schema(schema.schema_id, name="b")


def test_update_sparse_fields(store):
    exp_id = _create_experiments(store, "test_sparse")
    schema = _create_pass_fail_schema(store, exp_id)

    updated = store.update_label_schema(schema.schema_id, instruction="Updated instruction")
    assert updated.instruction == "Updated instruction"
    # Unchanged fields preserved
    assert updated.name == schema.name
    assert updated.input == schema.input


def test_update_input_replace(store):
    exp_id = _create_experiments(store, "test_input_replace")
    schema = _create_pass_fail_schema(store, exp_id)

    new_input = InputPassFail(positive_label="Good", negative_label="Bad")
    updated = store.update_label_schema(schema.schema_id, input=new_input)
    assert updated.input.positive_label == "Good"
    assert updated.input.negative_label == "Bad"


def test_update_rejects_input_variant_change(store):
    exp_id = _create_experiments(store, "test_variant_immutable")
    schema = _create_pass_fail_schema(store, exp_id)
    with pytest.raises(MlflowException, match="input type cannot be changed"):
        store.update_label_schema(
            schema.schema_id, input=InputNumeric(min_value=1.0, max_value=5.0)
        )


def test_update_rejects_multi_select_change(store):
    exp_id = _create_experiments(store, "test_multi_select_immutable")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Severity",
        type="feedback",
        input=InputCategorical(options=["low", "high"], multi_select=False),
    )
    with pytest.raises(MlflowException, match="multi_select` cannot be changed"):
        store.update_label_schema(
            schema.schema_id,
            input=InputCategorical(options=["low", "high"], multi_select=True),
        )
    # Editing the option list (same variant + multi_select) is still allowed.
    updated = store.update_label_schema(
        schema.schema_id,
        input=InputCategorical(options=["low", "medium", "high"], multi_select=False),
    )
    assert updated.input.options == ["low", "medium", "high"]


def test_update_missing(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_label_schema("ls-does-not-exist", instruction="X")


def test_delete_removes(store):
    exp_id = _create_experiments(store, "test_delete")
    schema = _create_pass_fail_schema(store, exp_id)

    store.delete_label_schema(schema.schema_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema(schema.schema_id)


def test_delete_missing_is_noop(store):
    # Should not raise.
    store.delete_label_schema("ls-does-not-exist")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"input": InputNumeric(min_value=5.0, max_value=5.0)},
            "strictly less than",
        ),
        (
            {"input": InputNumeric(min_value=10.0, max_value=1.0)},
            "strictly less than",
        ),
        (
            {"input": InputCategorical(options=[])},
            "non-empty list",
        ),
        (
            {"input": InputCategorical(options=["dup", "dup"])},
            "deduplicated",
        ),
        (
            {"input": InputCategorical(options=["a" * 65])},
            "at most 64 characters",
        ),
        (
            {"input": InputCategorical(options=[f"o{i}" for i in range(11)])},
            "at most 10",
        ),
        (
            {"input": InputCategorical(options=["a"], multi_select=1)},
            "must be a bool",
        ),
        (
            {"input": InputNumeric(min_value=True, max_value=5.0)},
            "must be numeric",
        ),
        (
            {"name": "a" * 251},
            "at most 250 characters",
        ),
        (
            {"instruction": "a" * 1001},
            "at most 1000 characters",
        ),
        (
            {"input": InputPassFail(positive_label="same", negative_label="same")},
            "must be distinct",
        ),
        (
            {"input": InputPassFail(positive_label="a" * 65, negative_label="b")},
            "at most 64",
        ),
        (
            {"input": InputNumeric(min_value="not-a-number", max_value=10)},
            "must be numeric",
        ),
        (
            {"input": InputText(max_length=0)},
            "at least 1",
        ),
        (
            {"enable_comment": "yes"},
            "must be a bool",
        ),
    ],
)
def test_create_validation_rejects_bad_inputs(store, kwargs, match):
    exp_id = _create_experiments(store, f"test_validation_{abs(hash(match))}")
    defaults = {
        "experiment_id": exp_id,
        "name": "valid_name",
        "type": "feedback",
        "input": InputPassFail(positive_label="Pass", negative_label="Fail"),
    }
    with pytest.raises(MlflowException, match=match):
        store.create_label_schema(**(defaults | kwargs))


def test_round_trip_preserves_categorical_multi_select(store):
    exp_id = _create_experiments(store, "test_round_trip_multi")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="Which tags apply?",
        type="feedback",
        input=InputCategorical(options=["bug", "feature", "ux"], multi_select=True),
    )
    fetched = store.get_label_schema(schema.schema_id)
    assert fetched.input.multi_select is True
    assert fetched.input.options == ["bug", "feature", "ux"]


def test_experiment_hard_delete_cascades(store):
    exp_id = _create_experiments(store, "test_cascade")
    schema = _create_pass_fail_schema(store, exp_id)
    assert store.get_label_schema(schema.schema_id).schema_id == schema.schema_id

    # Soft-delete alone does not fire the FK cascade; the row only goes away
    # when the experiment is hard-deleted.
    store.delete_experiment(exp_id)
    store._hard_delete_experiment(exp_id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema(schema.schema_id)


def test_label_schemas_blocked_when_experiment_soft_deleted(store):
    exp_id = _create_experiments(store, "test_soft_delete")
    schema = _create_pass_fail_schema(store, exp_id)

    store.delete_experiment(exp_id)

    # The pre-existing schema row is unchanged (soft-delete doesn't cascade).
    assert store.get_label_schema(schema.schema_id).schema_id == schema.schema_id

    # But new writes against the soft-deleted experiment are rejected.
    with pytest.raises(MlflowException, match="No Experiment with id"):
        _create_pass_fail_schema(store, exp_id, name="another")
