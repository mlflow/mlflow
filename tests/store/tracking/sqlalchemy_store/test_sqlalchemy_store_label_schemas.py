
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


def _create_pass_fail_schema(store, experiment_id, name="correctness"):
    return store.create_label_schema(
        experiment_id=experiment_id,
        name=name,
        type="feedback",
        title="Is the answer correct?",
        input=InputPassFail(positive_label="Correct", negative_label="Incorrect"),
        instruction="Mark Correct if accurate.",
        enable_comment=True,
    )


def test_create_pass_fail_schema(store):
    exp_id = _create_experiments(store, "test_create_pass_fail")

    schema = _create_pass_fail_schema(store, exp_id)

    assert schema.schema_id.startswith(SqlLabelSchema.LABEL_SCHEMA_ID_PREFIX)
    assert schema.experiment_id == exp_id
    assert schema.name == "correctness"
    assert schema.type == LabelSchemaType.FEEDBACK
    assert schema.title == "Is the answer correct?"
    assert schema.instruction == "Mark Correct if accurate."
    assert schema.enable_comment is True
    assert isinstance(schema.input, InputPassFail)
    assert schema.input.positive_label == "Correct"
    assert schema.input.negative_label == "Incorrect"
    assert schema.created_at > 0
    assert schema.updated_at == schema.created_at


def test_create_categorical_schema_requires_polarity_for_feedback(store):
    exp_id = _create_experiments(store, "test_categorical_feedback")
    with pytest.raises(MlflowException, match="semantic_polarity"):
        store.create_label_schema(
            experiment_id=exp_id,
            name="severity",
            type="feedback",
            title="Severity",
            input=InputCategorical(options=["low", "high"]),
        )


def test_create_categorical_schema_with_polarity(store):
    exp_id = _create_experiments(store, "test_categorical_polarity")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="severity",
        type="feedback",
        title="Severity",
        input=InputCategorical(
            options=["low", "medium", "high"],
            semantic_polarity="ascending",
            multi_select=False,
        ),
    )
    assert isinstance(schema.input, InputCategorical)
    assert schema.input.options == ["low", "medium", "high"]
    assert schema.input.semantic_polarity == "ascending"
    assert schema.input.multi_select is False


def test_create_numeric_schema_requires_bounds_for_feedback(store):
    exp_id = _create_experiments(store, "test_numeric_feedback")
    with pytest.raises(MlflowException, match="min_value.*max_value"):
        store.create_label_schema(
            experiment_id=exp_id,
            name="rating",
            type="feedback",
            title="Rating",
            input=InputNumeric(),
        )


def test_create_rejects_text_input_for_oss(store):
    exp_id = _create_experiments(store, "test_text_unsupported")
    with pytest.raises(MlflowException, match="not supported by the OSS server"):
        store.create_label_schema(
            experiment_id=exp_id,
            name="comment",
            type="feedback",
            title="Comment",
            input=InputText(max_length=500),
        )


def test_create_rejects_invalid_name(store):
    exp_id = _create_experiments(store, "test_invalid_name")
    with pytest.raises(MlflowException, match="alphanumeric and underscore"):
        store.create_label_schema(
            experiment_id=exp_id,
            name="bad-name-with-hyphens",
            type="feedback",
            title="X",
            input=InputPassFail(positive_label="a", negative_label="b"),
        )


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
            title="X",
            input=InputPassFail(positive_label="a", negative_label="b"),
        )


def test_get_by_id_and_by_name(store):
    exp_id = _create_experiments(store, "test_get")
    schema = _create_pass_fail_schema(store, exp_id)

    by_id = store.get_label_schema(schema.schema_id)
    assert by_id.schema_id == schema.schema_id
    assert by_id.input == schema.input

    by_name = store.get_label_schema_by_name(exp_id, "correctness")
    assert by_name.schema_id == schema.schema_id


def test_get_by_id_missing(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema("ls-does-not-exist")


def test_list_orders_by_created_time_desc(store):
    exp_id = _create_experiments(store, "test_list")
    s1 = _create_pass_fail_schema(store, exp_id, name="first")
    s2 = _create_pass_fail_schema(store, exp_id, name="second")
    s3 = _create_pass_fail_schema(store, exp_id, name="third")

    schemas = store.list_label_schemas(exp_id)
    assert len(schemas) == 3
    # Most recent first
    assert schemas[0].name == "third"
    assert schemas[2].name == "first"
    _ = s1, s2, s3  # silence unused warning


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

    updated = store.update_label_schema(schema.schema_id, name="answer_correctness")
    assert updated.name == "answer_correctness"

    # Old name no longer findable
    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema_by_name(exp_id, "correctness")

    # New name resolves to same schema_id
    re_fetched = store.get_label_schema_by_name(exp_id, "answer_correctness")
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

    updated = store.update_label_schema(
        schema.schema_id, title="Updated title", instruction="Updated instruction"
    )
    assert updated.title == "Updated title"
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


def test_update_missing(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_label_schema("ls-does-not-exist", title="X")


def test_upsert_creates(store):
    exp_id = _create_experiments(store, "test_upsert_new")
    schema = store.upsert_label_schema(
        experiment_id=exp_id,
        name="quality",
        type="feedback",
        title="Quality",
        input=InputNumeric(min_value=1.0, max_value=5.0),
    )
    assert schema.schema_id.startswith(SqlLabelSchema.LABEL_SCHEMA_ID_PREFIX)
    assert schema.input == InputNumeric(min_value=1.0, max_value=5.0)


def test_upsert_replaces(store):
    exp_id = _create_experiments(store, "test_upsert_replace")
    schema = _create_pass_fail_schema(store, exp_id)

    upserted = store.upsert_label_schema(
        experiment_id=exp_id,
        name=schema.name,
        type="feedback",
        title="Updated via upsert",
        input=InputPassFail(positive_label="Yes", negative_label="No"),
    )
    assert upserted.schema_id == schema.schema_id
    assert upserted.title == "Updated via upsert"
    assert upserted.input.positive_label == "Yes"


def test_upsert_rejects_type_change(store):
    exp_id = _create_experiments(store, "test_upsert_type_immutable")
    _create_pass_fail_schema(store, exp_id)

    with pytest.raises(MlflowException, match="type.*immutable"):
        store.upsert_label_schema(
            experiment_id=exp_id,
            name="correctness",
            type="expectation",
            title="x",
            input=InputPassFail(positive_label="a", negative_label="b"),
        )


def test_delete_removes(store):
    exp_id = _create_experiments(store, "test_delete")
    schema = _create_pass_fail_schema(store, exp_id)

    store.delete_label_schema(schema.schema_id)

    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema(schema.schema_id)


def test_delete_missing_is_noop(store):
    # Should not raise.
    store.delete_label_schema("ls-does-not-exist")


def test_round_trip_preserves_categorical_multi_select(store):
    exp_id = _create_experiments(store, "test_round_trip_multi")
    schema = store.create_label_schema(
        experiment_id=exp_id,
        name="applicable_tags",
        type="feedback",
        title="Which tags apply?",
        input=InputCategorical(
            options=["bug", "feature", "ux"],
            semantic_polarity="ascending",
            multi_select=True,
        ),
    )
    fetched = store.get_label_schema(schema.schema_id)
    assert fetched.input.multi_select is True
    assert fetched.input.options == ["bug", "feature", "ux"]


def test_experiment_delete_cascades(store):
    exp_id = _create_experiments(store, "test_cascade")
    schema = _create_pass_fail_schema(store, exp_id)
    assert store.get_label_schema(schema.schema_id).schema_id == schema.schema_id

    store.delete_experiment(exp_id)

    # The schema row should be deleted via FK cascade. Even after the
    # tombstoned experiment is hard-deleted, the schema should not be
    # retrievable.
    store._hard_delete_experiment(exp_id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_label_schema(schema.schema_id)
