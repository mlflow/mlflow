import json

import pytest
from click.testing import CliRunner

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import store
from mlflow.cli.agent import commands as agent_commands

from tests.agent_playground._factories import make_assertion_row


@pytest.fixture
def client(db_uri):
    original = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(db_uri)
    yield MlflowClient(tracking_uri=db_uri)
    mlflow.set_tracking_uri(original)


@pytest.fixture
def experiment_id(client):
    return client.create_experiment("agent_playground_cli_test")


@pytest.fixture
def assertion_row():
    return make_assertion_row()


@pytest.fixture
def runner():
    return CliRunner()


# The ``client`` fixture already calls ``mlflow.set_tracking_uri(db_uri)``
# so the CLI (which uses the fluent global URI) reads the same backend
# as the in-process store helpers. No ``monkeypatch.setenv`` needed.


def test_list_empty_table(runner, experiment_id):
    result = runner.invoke(agent_commands, ["test", "list", "-x", experiment_id])
    assert result.exit_code == 0
    assert "No test cases found" in result.output


def test_list_shows_seeded_case_as_table(runner, experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    result = runner.invoke(agent_commands, ["test", "list", "-x", experiment_id])
    assert result.exit_code == 0
    assert assertion_row.test_case_id in result.output
    # Table column header is ``kind`` to match the JSON wire format
    # exactly (``expectations.kind``).
    assert "kind" in result.output
    assert "assertion" in result.output


def test_list_reads_experiment_id_from_envvar(runner, experiment_id, assertion_row, monkeypatch):
    # ``mlflow agent test list`` registers ``MLFLOW_EXPERIMENT_ID`` as
    # the env-var fallback for ``-x``; verify that an invocation
    # without ``-x`` resolves the experiment from the env. A renaming
    # of the env var would surface in CI via this test.
    store.insert_case(experiment_id, assertion_row)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment_id)
    result = runner.invoke(agent_commands, ["test", "list"])
    assert result.exit_code == 0
    assert assertion_row.test_case_id in result.output


def test_list_json_output(runner, experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    result = runner.invoke(
        agent_commands, ["test", "list", "-x", experiment_id, "--output", "json"]
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 1
    assert parsed[0]["test_case_id"] == assertion_row.test_case_id


def test_delete_removes_case(runner, experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    result = runner.invoke(
        agent_commands,
        ["test", "delete", assertion_row.test_case_id, "-x", experiment_id, "-y"],
    )
    assert result.exit_code == 0
    assert store.get_case(experiment_id, assertion_row.test_case_id) is None


def test_delete_aborts_without_yes(runner, experiment_id, assertion_row):
    store.insert_case(experiment_id, assertion_row)
    result = runner.invoke(
        agent_commands,
        ["test", "delete", assertion_row.test_case_id, "-x", experiment_id],
        input="n\n",
    )
    assert result.exit_code != 0
    assert store.get_case(experiment_id, assertion_row.test_case_id) is not None


def test_delete_returns_error_for_missing(runner, experiment_id):
    result = runner.invoke(
        agent_commands, ["test", "delete", "tc-missing", "-x", experiment_id, "-y"]
    )
    assert result.exit_code != 0
    assert "not found" in result.output
