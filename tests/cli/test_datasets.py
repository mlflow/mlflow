import json
import os
import sys

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli.datasets import commands
from mlflow.genai.datasets import create_dataset


@pytest.fixture
def runner():
    return CliRunner(catch_exceptions=False)


@pytest.fixture(autouse=True)
def tracking_uri(tmp_path):
    if "MLFLOW_SKINNY" in os.environ:
        pytest.skip("SQLAlchemy store is not available in skinny.")

    original_tracking_uri = mlflow.get_tracking_uri()

    path = tmp_path.joinpath("mlflow.db").as_uri()
    tracking_uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[
        len("file://") :
    ]

    mlflow.set_tracking_uri(tracking_uri)

    yield tracking_uri

    mlflow.set_tracking_uri(original_tracking_uri)


@pytest.fixture
def experiment():
    exp_id = mlflow.create_experiment("test_datasets_cli")
    yield exp_id
    mlflow.delete_experiment(exp_id)


@pytest.fixture
def dataset_a(experiment):
    return create_dataset(
        name="dataset_a",
        experiment_id=experiment,
        tags={"env": "production"},
    )


@pytest.fixture
def dataset_b(experiment):
    return create_dataset(
        name="dataset_b",
        experiment_id=experiment,
        tags={"env": "staging"},
    )


def test_commands_group_exists():
    assert commands.name == "datasets"
    assert commands.help is not None


def test_list_command_params():
    list_cmd = next((cmd for cmd in commands.commands.values() if cmd.name == "list"), None)
    assert list_cmd is not None
    param_names = {p.name for p in list_cmd.params}
    expected_params = {
        "experiment_id",
        "filter_string",
        "max_results",
        "order_by",
        "page_token",
        "output",
    }
    assert param_names == expected_params


def test_list_datasets_table_output(runner: CliRunner, experiment: str, dataset_a):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment])

    assert result.exit_code == 0
    assert dataset_a.dataset_id in result.output
    assert "dataset_a" in result.output


def test_list_datasets_json_output(runner: CliRunner, experiment: str, dataset_a):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--output", "json"])

    assert result.exit_code == 0

    expected = {
        "datasets": [
            {
                "dataset_id": dataset_a.dataset_id,
                "name": "dataset_a",
                "digest": dataset_a.digest,
                "created_time": dataset_a.created_time,
                "last_update_time": dataset_a.last_update_time,
                "created_by": dataset_a.created_by,
                "last_updated_by": dataset_a.last_updated_by,
                "tags": dataset_a.tags,
            }
        ],
        "next_page_token": None,
    }
    assert json.loads(result.output) == expected


def test_list_datasets_empty_results(runner: CliRunner, experiment: str):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment])

    assert result.exit_code == 0


def test_list_datasets_json_empty_results(runner: CliRunner, experiment: str):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--output", "json"])

    assert result.exit_code == 0
    output_json = json.loads(result.output)
    assert output_json == {"datasets": [], "next_page_token": None}


def test_list_datasets_with_experiment_id_env_var(runner: CliRunner, experiment: str, dataset_a):
    result = runner.invoke(commands, ["list"], env={"MLFLOW_EXPERIMENT_ID": experiment})

    assert result.exit_code == 0
    assert dataset_a.dataset_id in result.output


def test_list_datasets_missing_experiment_id(runner: CliRunner):
    result = runner.invoke(commands, ["list"])

    assert result.exit_code != 0
    assert "Missing option '--experiment-id' / '-x'" in result.output


def test_list_datasets_invalid_output_format(runner: CliRunner, experiment: str):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--output", "invalid"])

    assert result.exit_code != 0
    assert "'invalid' is not one of 'table', 'json'" in result.output


def test_list_datasets_with_filter_string(runner: CliRunner, experiment: str, dataset_a, dataset_b):
    result = runner.invoke(
        commands,
        ["list", "--experiment-id", experiment, "--filter-string", "name = 'dataset_a'"],
    )

    assert result.exit_code == 0
    assert "dataset_a" in result.output
    assert "dataset_b" not in result.output


def test_list_datasets_with_max_results(runner: CliRunner, experiment: str, dataset_a, dataset_b):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--max-results", "1"])

    assert result.exit_code == 0
    output_lines = [line for line in result.output.split("\n") if "dataset_" in line]
    assert len(output_lines) == 1


def test_list_datasets_with_order_by(runner: CliRunner, experiment: str, dataset_a, dataset_b):
    result = runner.invoke(
        commands, ["list", "--experiment-id", experiment, "--order-by", "name ASC"]
    )

    assert result.exit_code == 0
    a_pos = result.output.find("dataset_a")
    b_pos = result.output.find("dataset_b")
    assert a_pos < b_pos


def test_list_datasets_short_option_x(runner: CliRunner, experiment: str, dataset_a):
    result = runner.invoke(commands, ["list", "-x", experiment])

    assert result.exit_code == 0
    assert dataset_a.dataset_id in result.output


def test_list_datasets_multiple_datasets(runner: CliRunner, experiment: str, dataset_a, dataset_b):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment])

    assert result.exit_code == 0
    assert dataset_a.dataset_id in result.output
    assert "dataset_a" in result.output
    assert dataset_b.dataset_id in result.output
    assert "dataset_b" in result.output


def test_list_datasets_json_multiple_datasets(
    runner: CliRunner, experiment: str, dataset_a, dataset_b
):
    result = runner.invoke(commands, ["list", "--experiment-id", experiment, "--output", "json"])

    assert result.exit_code == 0
    output_json = json.loads(result.output)
    assert len(output_json["datasets"]) == 2

    names = {d["name"] for d in output_json["datasets"]}
    assert names == {"dataset_a", "dataset_b"}
