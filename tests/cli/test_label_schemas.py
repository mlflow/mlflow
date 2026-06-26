import json
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli.label_schemas import commands
from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas import InputCategorical, InputPassFail, LabelSchema, LabelSchemaType
from mlflow.store.entities.paged_list import PagedList


@pytest.fixture
def runner():
    return CliRunner()


def _schema(name="correctness", schema_id="ls-1"):
    return LabelSchema(
        name=name,
        type=LabelSchemaType.FEEDBACK,
        input=InputPassFail(positive_label="Correct", negative_label="Wrong"),
        instruction=None,
        enable_comment=False,
        schema_id=schema_id,
        experiment_id="0",
    )


def test_create_passfail_json(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.create_label_schema", return_value=_schema()
    ) as mock_create:
        result = runner.invoke(
            commands,
            [
                "create",
                "--name",
                "correctness",
                "--type",
                "feedback",
                "--input",
                '{"variant": "pass_fail", "positive_label": "Correct", "negative_label": "Wrong"}',
                "--experiment-id",
                "0",
                "--output",
                "json",
            ],
        )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_id"] == "ls-1"
    assert payload["name"] == "correctness"
    mock_create.assert_called_once_with(
        name="correctness",
        type="feedback",
        input=InputPassFail(positive_label="Correct", negative_label="Wrong"),
        instruction=None,
        enable_comment=False,
        experiment_id="0",
    )


def test_create_categorical_input_parsed(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.create_label_schema", return_value=_schema()
    ) as mock_create:
        result = runner.invoke(
            commands,
            [
                "create",
                "--name",
                "tone",
                "--type",
                "feedback",
                "--input",
                '{"variant": "categorical", "options": ["a", "b"], "multi_select": true}',
                "--experiment-id",
                "0",
            ],
        )
    assert result.exit_code == 0
    _, kwargs = mock_create.call_args
    assert kwargs["input"] == InputCategorical(options=["a", "b"], multi_select=True)


def test_create_requires_experiment_id(runner):
    result = runner.invoke(
        commands,
        [
            "create",
            "--name",
            "x",
            "--type",
            "feedback",
            "--input",
            '{"variant": "text"}',
        ],
        env={"MLFLOW_EXPERIMENT_ID": ""},
    )
    assert result.exit_code == 2
    assert "--experiment-id is required" in result.output


def test_create_bad_input_variant(runner):
    result = runner.invoke(
        commands,
        [
            "create",
            "--name",
            "x",
            "--type",
            "feedback",
            "--input",
            '{"variant": "slider"}',
            "--experiment-id",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "variant must be one of" in result.output


def test_create_bad_type_enum(runner):
    result = runner.invoke(
        commands,
        ["create", "--name", "x", "--type", "bogus", "--input", '{"variant": "text"}', "-x", "0"],
    )
    assert result.exit_code == 2
    assert "'bogus' is not one of 'feedback', 'expectation'" in result.output


@pytest.mark.parametrize("args", [[], ["--schema-id", "ls-1", "--name", "n"]])
def test_get_xor_selector(runner, args):
    result = runner.invoke(commands, ["get", *args, "-x", "0"])
    assert result.exit_code == 2
    assert "exactly one of --schema-id or --name" in result.output


def test_get_by_id(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.get_label_schema", return_value=_schema()
    ) as mock_get:
        result = runner.invoke(commands, ["get", "--schema-id", "ls-1", "--output", "json"])
    assert result.exit_code == 0
    mock_get.assert_called_once_with(name=None, schema_id="ls-1", experiment_id=None)


def test_list_default_max_results(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.list_label_schemas",
        return_value=PagedList([_schema()], None),
    ) as mock_list:
        result = runner.invoke(commands, ["list", "-x", "0", "--output", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["next_page_token"] is None
    mock_list.assert_called_once_with(experiment_id="0", max_results=100, page_token=None)


def test_list_pagetoken_teachline(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.list_label_schemas",
        return_value=PagedList([_schema()], "tok-123"),
    ):
        result = runner.invoke(commands, ["list", "-x", "0"])
    assert result.exit_code == 0
    assert "--page-token tok-123" in result.stderr


def test_update_sparse_leaves_enable_comment_none(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.update_label_schema", return_value=_schema()
    ) as mock_update:
        result = runner.invoke(commands, ["update", "--schema-id", "ls-1", "--name", "new"])
    assert result.exit_code == 0
    mock_update.assert_called_once_with(
        schema_id="ls-1", name="new", instruction=None, enable_comment=None, input=None
    )


def test_delete_json_confirmation(runner):
    with mock.patch("mlflow.cli.label_schemas.delete_label_schema") as mock_delete:
        result = runner.invoke(commands, ["delete", "--schema-id", "ls-1", "--output", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"schema_id": "ls-1", "deleted": True}
    mock_delete.assert_called_once_with(schema_id="ls-1")


def test_mlflow_exception_rendered_cleanly(runner):
    with mock.patch(
        "mlflow.cli.label_schemas.get_label_schema",
        side_effect=MlflowException("not supported on a Databricks tracking URI"),
    ):
        result = runner.invoke(commands, ["get", "--schema-id", "ls-1"])
    assert result.exit_code == 1
    assert "not supported on a Databricks tracking URI" in result.output
    assert "Traceback" not in result.output
