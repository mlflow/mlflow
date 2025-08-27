import json
import os
import textwrap
from unittest import mock

import pytest
from click.testing import CliRunner

import mlflow
from mlflow import experiments
from mlflow.runs import create_run, list_run


def test_list_run():
    with mlflow.start_run(run_name="apple"):
        pass
    result = CliRunner().invoke(list_run, ["--experiment-id", "0"])
    assert "apple" in result.output


def test_list_run_experiment_id_required():
    result = CliRunner().invoke(list_run, [])
    assert "Missing option '--experiment-id'" in result.output


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny Client does not support predict due to the pandas dependency",
)
def test_csv_generation(tmp_path):
    import numpy as np
    import pandas as pd

    with mock.patch(
        "mlflow.experiments.fluent.search_runs",
        return_value=pd.DataFrame(
            {
                "run_id": np.array(["all_set", "with_none", "with_nan"]),
                "experiment_id": np.array([1, 1, 1]),
                "param_optimizer": np.array(["Adam", None, "Adam"]),
                "avg_loss": np.array([42.0, None, np.nan], dtype=np.float32),
            },
            columns=["run_id", "experiment_id", "param_optimizer", "avg_loss"],
        ),
    ):
        expected_csv = textwrap.dedent(
            """\
        run_id,experiment_id,param_optimizer,avg_loss
        all_set,1,Adam,42.0
        with_none,1,,
        with_nan,1,Adam,
        """
        )
        result_filename = os.path.join(tmp_path, "result.csv")
        CliRunner().invoke(
            experiments.generate_csv_with_runs,
            ["--experiment-id", "1", "--filename", result_filename],
        )
        with open(result_filename) as fd:
            assert expected_csv == fd.read()


def test_create_run_with_experiment_id():
    mlflow.create_experiment("test_create_run_exp")
    exp = mlflow.get_experiment_by_name("test_create_run_exp")

    result = CliRunner().invoke(create_run, ["--experiment-id", exp.experiment_id])
    assert result.exit_code == 0

    output = json.loads(result.output)
    assert "run_id" in output
    assert output["experiment_id"] == exp.experiment_id
    assert output["status"] == "FINISHED"

    # Verify the run was created
    run = mlflow.get_run(output["run_id"])
    assert run.info.experiment_id == exp.experiment_id
    assert run.info.status == "FINISHED"


def test_create_run_with_experiment_name():
    exp_name = "test_create_run_by_name"

    result = CliRunner().invoke(create_run, ["--experiment-name", exp_name])
    assert result.exit_code == 0

    # Extract JSON from output (it may be mixed with logs)
    lines = result.output.strip().split("\n")
    try:
        # Find lines that contain just the JSON brackets
        start = next(i for i, line in enumerate(lines) if line.strip() == "{")
        end = next(i for i, line in enumerate(lines) if line.strip() == "}")
        json_output = "\n".join(lines[start : end + 1])
        output = json.loads(json_output)
    except (StopIteration, ValueError):
        assert False, "No JSON output found"
    assert "run_id" in output
    assert output["status"] == "FINISHED"

    # Verify experiment was created
    exp = mlflow.get_experiment_by_name(exp_name)
    assert exp is not None
    assert output["experiment_id"] == exp.experiment_id


def test_create_run_with_custom_name_and_description():
    mlflow.create_experiment("test_run_with_details")
    exp = mlflow.get_experiment_by_name("test_run_with_details")

    run_name = "my-custom-run"
    description = "This is a test run"

    result = CliRunner().invoke(
        create_run,
        [
            "--experiment-id",
            exp.experiment_id,
            "--run-name",
            run_name,
            "--description",
            description,
        ],
    )
    assert result.exit_code == 0

    output = json.loads(result.output)
    assert output["run_name"] == run_name

    # Verify run details
    run = mlflow.get_run(output["run_id"])
    assert run.data.tags.get("mlflow.note.content") == description
    assert run.info.run_name == run_name


def test_create_run_with_tags():
    mlflow.create_experiment("test_run_with_tags")
    exp = mlflow.get_experiment_by_name("test_run_with_tags")

    result = CliRunner().invoke(
        create_run,
        [
            "--experiment-id",
            exp.experiment_id,
            "--tags",
            "env=test",
            "--tags",
            "model=linear",
            "--tags",
            "version=1.0",
        ],
    )
    assert result.exit_code == 0

    output = json.loads(result.output)
    run = mlflow.get_run(output["run_id"])

    assert run.data.tags["env"] == "test"
    assert run.data.tags["model"] == "linear"
    assert run.data.tags["version"] == "1.0"


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_create_run_with_different_status(status):
    mlflow.create_experiment("test_run_statuses")
    exp = mlflow.get_experiment_by_name("test_run_statuses")

    result = CliRunner().invoke(
        create_run, ["--experiment-id", exp.experiment_id, "--status", status]
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["status"] == status

    run = mlflow.get_run(output["run_id"])
    assert run.info.status == status


def test_create_run_missing_experiment():
    result = CliRunner().invoke(create_run, [])
    assert result.exit_code != 0
    assert "Must specify exactly one of --experiment-id or --experiment-name" in result.output


def test_create_run_both_experiment_params():
    result = CliRunner().invoke(create_run, ["--experiment-id", "0", "--experiment-name", "test"])
    assert result.exit_code != 0
    assert "Must specify exactly one of --experiment-id or --experiment-name" in result.output


def test_create_run_invalid_tag_format():
    mlflow.create_experiment("test_invalid_tag")
    exp = mlflow.get_experiment_by_name("test_invalid_tag")

    result = CliRunner().invoke(
        create_run, ["--experiment-id", exp.experiment_id, "--tags", "invalid-tag"]
    )
    assert result.exit_code != 0
    assert "Invalid tag format" in result.output


def test_create_run_duplicate_tag_key():
    mlflow.create_experiment("test_duplicate_tag")
    exp = mlflow.get_experiment_by_name("test_duplicate_tag")

    result = CliRunner().invoke(
        create_run,
        ["--experiment-id", exp.experiment_id, "--tags", "env=test", "--tags", "env=prod"],
    )
    assert result.exit_code != 0
    assert "Duplicate tag key" in result.output
