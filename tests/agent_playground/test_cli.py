import json

import pytest
from click.testing import CliRunner

import mlflow
from mlflow import MlflowClient
from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import AssertionSpec, TestSpec
from mlflow.cli.agent import commands as agent_commands


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
def assertion_spec():
    return TestSpec(
        strategy="assertion",
        rationale_summary="agent must cite docs",
        assertion=AssertionSpec(must_contain=["docs"]),
    )


@pytest.fixture
def runner():
    return CliRunner()


def test_list_empty_table(runner, experiment_id, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    result = runner.invoke(agent_commands, ["test", "list", "-x", experiment_id])
    assert result.exit_code == 0
    assert "No test cases found" in result.output


def test_list_shows_seeded_case_as_table(
    runner, experiment_id, assertion_spec, db_uri, monkeypatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    result = runner.invoke(agent_commands, ["test", "list", "-x", experiment_id])
    assert result.exit_code == 0
    assert test_case_id in result.output
    assert "assertion" in result.output


def test_list_json_output(runner, experiment_id, assertion_spec, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    result = runner.invoke(
        agent_commands, ["test", "list", "-x", experiment_id, "--output", "json"]
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 1
    assert parsed[0]["test_case_id"] == test_case_id


def test_delete_removes_case(runner, experiment_id, assertion_spec, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    result = runner.invoke(
        agent_commands, ["test", "delete", test_case_id, "-x", experiment_id, "-y"]
    )
    assert result.exit_code == 0
    assert store.get_case(experiment_id, test_case_id) is None


def test_delete_aborts_without_yes(runner, experiment_id, assertion_spec, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    test_case_id = store.insert_case(experiment_id, assertion_spec)
    result = runner.invoke(
        agent_commands,
        ["test", "delete", test_case_id, "-x", experiment_id],
        input="n\n",
    )
    assert result.exit_code != 0
    assert store.get_case(experiment_id, test_case_id) is not None


def test_delete_returns_error_for_missing(runner, experiment_id, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    result = runner.invoke(
        agent_commands, ["test", "delete", "tc-missing", "-x", experiment_id, "-y"]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_export_to_stdout(runner, experiment_id, assertion_spec, db_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    test_case_id = store.insert_case(
        experiment_id,
        assertion_spec,
        conversation_messages=[{"role": "user", "content": "hi"}],
    )
    result = runner.invoke(agent_commands, ["test", "export", "-x", experiment_id])
    assert result.exit_code == 0
    assert test_case_id in result.output
    assert "MLFLOW_AGENT_URL" in result.output


def test_export_to_file(runner, experiment_id, assertion_spec, db_uri, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db_uri)
    test_case_id = store.insert_case(
        experiment_id,
        assertion_spec,
        conversation_messages=[{"role": "user", "content": "hi"}],
    )
    out = tmp_path / "test_suite.py"
    result = runner.invoke(agent_commands, ["test", "export", "-x", experiment_id, "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text()
    assert test_case_id in content
    assert "Wrote pytest suite" in result.output
