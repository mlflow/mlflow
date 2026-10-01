import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

import issue_safety as safety


@pytest.mark.parametrize(
    ("title", "body"),
    [
        ("-----BEGIN PRIVATE KEY-----", ""),
        ("A bug", "password = verySecret123"),
        ("ghp_" + "a" * 36, ""),
        ("A bug", "AKIA" + "A" * 16),
        ("A bug", "api_key: sk-" + "a" * 25),
        ("A bug", "Authorization: Bearer " + "a" * 32),
        ("A bug", "AWS_SECRET_ACCESS_KEY=" + "a" * 40),
    ],
)
def test_preflight_blocks_secrets_before_cleaning(title, body):
    assert safety.preflight(title, body) == ("unsafe", "", "")


def test_preflight_cleans_comments_and_controls_but_preserves_evidence():
    assert safety.preflight(
        "Bug<!-- hidden\ninstructions -->\x00",
        "```python\nprint(1)\n```\x1b\nhttps://example.com/logs",
    ) == (None, "Bug", "```python\nprint(1)\n```\nhttps://example.com/logs")


@pytest.mark.parametrize(
    ("title", "body"),
    [("x" * 513, ""), ("title", "x" * 16_001)],
)
def test_preflight_rejects_large_cleaned_inputs(title, body):
    assert safety.preflight(title, body) == ("uncertain", "", "")


def response(decision="safe", rationale="Looks suitable."):
    return {
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": json.dumps({"decision": decision, "rationale": rationale})}
        ],
    }


@pytest.mark.parametrize(
    "invalid",
    [
        {"content": response()["content"]},
        {"stop_reason": "max_tokens", "content": response()["content"]},
        {"stop_reason": "end_turn", "content": []},
        {"stop_reason": "end_turn", "content": response()["content"] * 2},
        {"stop_reason": "end_turn", "content": [{"type": "tool_use", "text": "{}"}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "not json"}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"decision":"safe"}'}]},
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"decision":"safe","rationale":"ok","extra":1}'}],
        },
        response("unknown"),
        response("safe", ""),
        response("safe", "x" * 241),
    ],
)
def test_validate_response_rejects_invalid_output(invalid):
    with pytest.raises((ValueError, TypeError)):
        safety.validate_response(invalid)


def test_validate_response_accepts_contract():
    assert safety.validate_response(response()) == ("safe", "Looks suitable.")


def test_assess_uses_tagged_anthropic_messages_without_real_gateway(monkeypatch, capsys):
    monkeypatch.setenv("DATABRICKS_GATEWAY_HOST", "https://example.databricks.com")
    monkeypatch.setenv("DATABRICKS_GATEWAY_CLIENT_ID", "fake-client")
    monkeypatch.setenv("DATABRICKS_GATEWAY_CLIENT_SECRET", "fake-secret")
    monkeypatch.setenv("GITHUB_REPOSITORY", "mlflow/mlflow")
    with mock.patch.object(
        safety, "_read_json", side_effect=[{"access_token": "fake-bearer"}, response()]
    ) as read_json:
        assert safety.assess("title", "body") == ("safe", "Looks suitable.")
        assert read_json.call_count == 2
    auth_request, model_request = (call.args[0] for call in read_json.call_args_list)
    assert auth_request.full_url == "https://example.databricks.com/oidc/v1/token"
    assert (
        model_request.full_url == "https://example.databricks.com/ai-gateway/anthropic/v1/messages"
    )
    assert model_request.get_header("Authorization").startswith("Bearer ")
    assert model_request.get_header("Authorization").endswith("fake-bearer")
    assert json.loads(model_request.get_header("Databricks-ai-gateway-request-tags")) == {
        "repository": "mlflow/mlflow",
        "task": "issue-triage-safety",
    }
    payload = json.loads(model_request.data)
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["output_config"]["format"]["schema"] == safety.SCHEMA
    assert json.loads(payload["messages"][0]["content"]) == {"title": "title", "body": "body"}
    assert capsys.readouterr().out == "::add-mask::fake-bearer\n"


@pytest.mark.parametrize("decision", ["safe", "unsafe", "uncertain"])
def test_main_only_creates_safe_artifact(tmp_path, monkeypatch, decision):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({
            "issue": {
                "number": 123,
                "title": "Bug<!-- hide -->",
                "body": "logs:\x1b\nhttps://example.com",
            },
        })
    )
    summary = tmp_path / "summary"
    context = tmp_path / "context.json"
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_EVENT_PATH": event,
        "GITHUB_STEP_SUMMARY": summary,
        "CONTEXT_PATH": context,
        "GITHUB_OUTPUT": output,
        "GITHUB_REPOSITORY": "mlflow/mlflow",
    }.items():
        monkeypatch.setenv(key, str(value))
    with mock.patch.object(
        safety, "assess", return_value=(decision, "<b>ok</b>\n::warning::")
    ) as assess:
        assert safety.main() == 0
        assess.assert_called_once_with("Bug", "logs:\nhttps://example.com")
    assert "&lt;b&gt;ok&lt;/b&gt; ::warning::" in summary.read_text()
    if decision == "safe":
        assert json.loads(context.read_text()) == {
            "title": "Bug",
            "body": "logs:\nhttps://example.com",
            "repository": "mlflow/mlflow",
            "issue_number": 123,
        }
        assert output.read_text() == "context_ready=true\n"
    else:
        assert not context.exists()
        assert not output.exists()


def test_main_does_not_call_model_on_preflight_rejection(tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"issue": {"number": 1, "title": "Bug", "body": None}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    with mock.patch.object(safety, "assess", return_value=("uncertain", "No details")) as assess:
        assert safety.main() == 0
        assess.assert_called_once_with("Bug", "")
    event.write_text(
        json.dumps({"issue": {"number": 1, "title": "password: secret1234", "body": ""}})
    )
    with mock.patch.object(safety, "assess") as assess:
        assert safety.main() == 0
        assess.assert_not_called()


def test_main_fails_closed_on_gateway_error(tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"issue": {"number": 1, "title": "Bug", "body": ""}}))
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("CONTEXT_PATH", str(tmp_path / "context.json"))
    with mock.patch.object(safety, "assess", side_effect=RuntimeError("sensitive data")) as assess:
        assert safety.main() == 1
        assess.assert_called_once_with("Bug", "")
    assert "uncertain" in summary.read_text()
    assert "sensitive data" not in summary.read_text()
    assert not (tmp_path / "context.json").exists()


def test_workflow_handoff_and_shell_safety():
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/triage.yml").read_text()
    )
    jobs = workflow["jobs"]
    assert jobs["label"]["needs"] == "security_gate"
    assert jobs["safety"]["needs"] == "security_gate"
    assert jobs["safety"]["if"] == "needs.security_gate.outputs.security_match == 'false'"
    assert jobs["triage_smoke"]["needs"] == "safety"
    assert jobs["triage_smoke"]["if"] == "needs.safety.outputs.context_ready == 'true'"
    assert jobs["safety"]["permissions"] == {"contents": "read"}
    assert "permissions" not in jobs["triage_smoke"]
    assert "jq -c '{title, body}' \"$CONTEXT_PATH\"" in jobs["triage_smoke"]["steps"][-1]["run"]
    assert jobs["safety"]["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    artifact_name = jobs["safety"]["steps"][-1]["with"]["name"]
    assert artifact_name == jobs["triage_smoke"]["steps"][0]["with"]["name"]
    assert "${{ github.run_attempt }}" in artifact_name
    assert "${{ github.event.issue.title }}" not in str(jobs["safety"])
    assert "${{ github.event.issue.body }}" not in str(jobs["safety"])


def test_smoke_compact_json_keeps_workflow_commands_within_one_line(tmp_path):
    context = tmp_path / "issue-context.json"
    context.write_text(
        json.dumps({
            "title": "Bug\n::error::injected",
            "body": "line\r\n::add-mask::injected",
        })
    )
    result = subprocess.run(
        ["jq", "-c", "{title, body}", context], capture_output=True, text=True, check=True
    )
    assert result.stdout.count("\n") == 1
    assert result.stdout.startswith('{"title":')
    assert json.loads(result.stdout)["title"] == "Bug\n::error::injected"
