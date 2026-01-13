import json
import logging
import os
import textwrap
from unittest import mock
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import mlflow
from mlflow import experiments
from mlflow.exceptions import MlflowException
from mlflow.runs import create_run, link_traces, list_run


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress logging for all tests to ensure clean CLI output."""
    # Suppress all logging
    logging.disable(logging.CRITICAL)

    yield

    # Re-enable logging
    logging.disable(logging.NOTSET)


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

    output = json.loads(result.output)
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


def test_link_traces_single_trace():
    with patch("mlflow.runs.MlflowClient.link_traces_to_run") as mock_link_traces:
        result = CliRunner().invoke(
            link_traces,
            ["--run-id", "test_run_123", "--trace-id", "trace_abc"],
        )

        assert result.exit_code == 0
        assert "Successfully linked 1 trace(s) to run 'test_run_123'" in result.output
        mock_link_traces.assert_called_once_with(["trace_abc"], "test_run_123")


def test_link_traces_multiple_traces():
    with patch("mlflow.runs.MlflowClient.link_traces_to_run") as mock_link_traces:
        result = CliRunner().invoke(
            link_traces,
            [
                "--run-id",
                "test_run_456",
                "--trace-id",
                "trace_1",
                "--trace-id",
                "trace_2",
                "--trace-id",
                "trace_3",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully linked 3 trace(s) to run 'test_run_456'" in result.output
        mock_link_traces.assert_called_once_with(["trace_1", "trace_2", "trace_3"], "test_run_456")


def test_link_traces_with_short_option():
    with patch("mlflow.runs.MlflowClient.link_traces_to_run") as mock_link_traces:
        result = CliRunner().invoke(
            link_traces,
            ["--run-id", "run_789", "-t", "trace_x", "-t", "trace_y"],
        )

        assert result.exit_code == 0
        assert "Successfully linked 2 trace(s) to run 'run_789'" in result.output
        mock_link_traces.assert_called_once_with(["trace_x", "trace_y"], "run_789")


def test_link_traces_file_store_error():
    with patch(
        "mlflow.runs.MlflowClient.link_traces_to_run",
        side_effect=MlflowException(
            "Linking traces to runs is not supported in FileStore. "
            "Please use a database-backed store (e.g., SQLAlchemy store) for this feature."
        ),
    ):
        result = CliRunner().invoke(
            link_traces,
            ["--run-id", "test_run", "--trace-id", "trace_1"],
        )

        assert result.exit_code != 0
        assert "Failed to link traces" in result.output
        assert "not supported in FileStore" in result.output


def test_link_traces_too_many_traces_error():
    with patch(
        "mlflow.runs.MlflowClient.link_traces_to_run",
        side_effect=MlflowException(
            "Cannot link more than 100 traces to a run in a single request. Provided 101 traces."
        ),
    ):
        result = CliRunner().invoke(
            link_traces,
            ["--run-id", "test_run", "--trace-id", "trace_1"],
        )

        assert result.exit_code != 0
        assert "Failed to link traces" in result.output
        assert "100" in result.output


def test_link_traces_missing_run_id():
    result = CliRunner().invoke(link_traces, ["--trace-id", "trace_1"])

    assert result.exit_code != 0
    assert "Missing option '--run-id'" in result.output


def test_link_traces_missing_trace_id():
    result = CliRunner().invoke(link_traces, ["--run-id", "test_run"])

    assert result.exit_code != 0
    assert "Missing option '--trace-id'" in result.output


def test_link_traces_generic_error():
    with patch(
        "mlflow.runs.MlflowClient.link_traces_to_run",
        side_effect=MlflowException("Some other error"),
    ):
        result = CliRunner().invoke(
            link_traces,
            ["--run-id", "test_run", "--trace-id", "trace_1"],
        )

        assert result.exit_code != 0
        assert "Failed to link traces: Some other error" in result.output


def test_get_experiment_default():
    result = CliRunner().invoke(experiments.get_experiment, ["--experiment-id", "0"])
    assert result.exit_code == 0

    # Default output is table format
    assert "Experiment ID" in result.output
    assert "Name" in result.output
    assert "Artifact Location" in result.output
    assert "Lifecycle Stage" in result.output
    assert ":" in result.output


def test_get_experiment_json():
    exp_id = mlflow.create_experiment("test_get_exp_json", tags={"env": "test"})
    exp = mlflow.get_experiment(exp_id)

    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-id", exp_id, "--output", "json"]
    )
    assert result.exit_code == 0

    output = json.loads(result.output)
    expected = {
        "experiment_id": exp_id,
        "name": "test_get_exp_json",
        "artifact_location": exp.artifact_location,
        "lifecycle_stage": "active",
        "tags": {"env": "test"},
        "creation_time": exp.creation_time,
        "last_update_time": exp.last_update_time,
    }
    assert output == expected


def test_get_experiment_table():
    exp_id = mlflow.create_experiment("test_get_exp_table", tags={"env": "test", "team": "ml"})

    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-id", exp_id, "--output", "table"]
    )
    assert result.exit_code == 0

    # Verify table format
    assert "Experiment ID" in result.output
    assert exp_id in result.output
    assert "Name" in result.output
    assert "test_get_exp_table" in result.output
    assert "Lifecycle Stage" in result.output
    assert "active" in result.output
    assert "Tags" in result.output
    assert "env=test" in result.output
    assert "team=ml" in result.output


def test_get_experiment_table_no_tags():
    exp_id = mlflow.create_experiment("test_get_exp_no_tags")

    result = CliRunner().invoke(experiments.get_experiment, ["-x", exp_id, "--output", "table"])
    assert result.exit_code == 0

    assert "Experiment ID" in result.output
    assert exp_id in result.output
    assert "Tags" in result.output


def test_get_experiment_missing_id():
    result = CliRunner().invoke(experiments.get_experiment, [])
    assert result.exit_code != 0
    assert "Must specify exactly one of --experiment-id or --experiment-name" in result.output


def test_get_experiment_invalid_id():
    result = CliRunner().invoke(experiments.get_experiment, ["-x", "999999"])
    assert result.exit_code != 0


def test_get_experiment_deleted():
    exp_id = mlflow.create_experiment("test_deleted")
    mlflow.delete_experiment(exp_id)

    result = CliRunner().invoke(experiments.get_experiment, ["-x", exp_id, "--output", "json"])
    assert result.exit_code == 0

    output = json.loads(result.output)
    assert output["lifecycle_stage"] == "deleted"
    assert output["experiment_id"] == exp_id


def test_get_experiment_by_name_table():
    exp_name = "test_get_by_name"
    exp_id = mlflow.create_experiment(exp_name, tags={"env": "test"})

    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-name", exp_name, "--output", "table"]
    )
    assert result.exit_code == 0
    assert "Experiment ID" in result.output
    assert exp_id in result.output
    assert "Name" in result.output
    assert exp_name in result.output
    assert "Tags" in result.output
    assert "env=test" in result.output


def test_get_experiment_by_name_json():
    exp_name = "test_get_by_name_json"
    exp_id = mlflow.create_experiment(exp_name, tags={"team": "ml"})
    exp = mlflow.get_experiment(exp_id)

    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-name", exp_name, "--output", "json"]
    )
    assert result.exit_code == 0

    output = json.loads(result.output)
    expected = {
        "experiment_id": exp_id,
        "name": exp_name,
        "artifact_location": exp.artifact_location,
        "lifecycle_stage": "active",
        "tags": {"team": "ml"},
        "creation_time": exp.creation_time,
        "last_update_time": exp.last_update_time,
    }
    assert output == expected


def test_get_experiment_by_name_short_option():
    exp_name = "test_short_option"
    exp_id = mlflow.create_experiment(exp_name)

    result = CliRunner().invoke(experiments.get_experiment, ["-n", exp_name])
    assert result.exit_code == 0
    assert exp_id in result.output
    assert exp_name in result.output


def test_get_experiment_by_name_not_found():
    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-name", "nonexistent_experiment"]
    )
    assert result.exit_code != 0


def test_get_experiment_both_options_provided():
    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-id", "0", "--experiment-name", "Default"]
    )
    assert result.exit_code != 0
    assert "Must specify exactly one of --experiment-id or --experiment-name" in result.output


def test_get_experiment_by_name_deleted():
    exp_name = "test_deleted_by_name"
    exp_id = mlflow.create_experiment(exp_name)
    mlflow.delete_experiment(exp_id)

    result = CliRunner().invoke(
        experiments.get_experiment, ["--experiment-name", exp_name, "--output", "json"]
    )
    assert result.exit_code == 0

    output = json.loads(result.output)
    assert output["lifecycle_stage"] == "deleted"
    assert output["name"] == exp_name


def test_get_experiment_by_name_with_spaces():
    exp_name = "My Test Experiment"
    exp_id = mlflow.create_experiment(exp_name)

    result = CliRunner().invoke(experiments.get_experiment, ["--experiment-name", exp_name])
    assert result.exit_code == 0
    assert exp_id in result.output
    assert exp_name in result.output
