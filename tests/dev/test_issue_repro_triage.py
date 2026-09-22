import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

import issue_repro_triage as triage


class FakeAnthropicClient(triage.AnthropicClient):
    def __init__(self, response):
        super().__init__("secret")
        self.response = response
        self.body = None

    def _request(self, body):
        self.body = body
        return self.response


def test_anthropic_client_translates_exactly_one_tool_use():
    client = FakeAnthropicClient({
        "content": [
            {
                "type": "tool_use",
                "name": "read_tracked_file",
                "input": {"path": "mlflow/version.py"},
            }
        ]
    })

    result = client(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "issue data"},
        ],
        tools=[{"name": "read_tracked_file", "input_schema": {"type": "object"}}],
    )

    assert result == {"action": "read_tracked_file", "path": "mlflow/version.py"}
    assert client.body["system"] == "system"
    assert client.body["tool_choice"] == {"type": "any"}
    assert "output_config" not in client.body


def test_anthropic_client_returns_structured_text_without_tools():
    client = FakeAnthropicClient({"content": [{"type": "text", "text": '{"ok": true}'}]})

    result = client(
        messages=[{"role": "user", "content": "data"}],
        output_schema={"type": "object"},
    )

    assert result == '{"ok": true}'
    assert client.body["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in client.body


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "text", "text": "not a tool"}],
        [
            {"type": "tool_use", "name": "finish", "input": {}},
            {"type": "tool_use", "name": "finish", "input": {}},
        ],
        [{"type": "tool_use", "name": "finish", "input": {"action": "run_repro"}}],
    ],
)
def test_anthropic_client_rejects_ambiguous_or_reserved_tool_output(content):
    client = FakeAnthropicClient({"content": content})

    with pytest.raises(ValueError, match="Anthropic response|tool input"):
        client(
            messages=[{"role": "user", "content": "data"}],
            tools=[{"name": "finish", "input_schema": {"type": "object"}}],
        )


def _issue(labels=None):
    return {
        "number": 123,
        "state": "open",
        "title": "Call returns the wrong value",
        "body": "The call returns 2; it should return 1.",
        "labels": labels or [{"name": "bug"}],
    }


def _judgment(**updates):
    value = {
        "issue_kind": "bug",
        "surface": "python_core",
        "fidelity": {
            "verdict": "faithful",
            "confidence": 0.95,
            "failure_origin": "reported_symptom",
        },
        "proposed_fix": {
            "scope": "small",
            "confidence": 0.9,
        },
        "safety_uncertainty": False,
    }
    value.update(updates)
    return value


class JudgeClient:
    model = "judge-model"

    def __init__(self, judgment):
        self.judgment = judgment
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert "tools" not in kwargs
        return json.dumps(self.judgment)


def _install_broker(monkeypatch, tmp_path, *, failure=None):
    class Broker:
        def __init__(self, *, scratch_root, checkout_sha, **kwargs):
            self.scratch_root = Path(scratch_root)
            self.checkout_sha = checkout_sha
            (self.scratch_root / "reproduce.py").write_text(
                "import mlflow\nprint('actual symptom')\n", encoding="utf-8"
            )
            self.run_results = [
                {
                    "stdout": "trusted stdout\n",
                    "stderr": "trusted stderr\n",
                    "exit_status": 1,
                    "duration_seconds": 0.25,
                    "failure": failure,
                }
            ]

    monkeypatch.setattr(triage, "ReproductionBroker", Broker)
    monkeypatch.setattr(
        triage,
        "run_agent",
        lambda **kwargs: {
            "claimed_symptom": "The call returns 2 instead of 1.",
            "run_index": 0,
            "proposed_fix_summary": "Correct the localized return condition.",
            "environment_limitations": [],
        },
    )


def test_investigation_runs_one_tools_less_judge_and_binds_trusted_evidence(monkeypatch, tmp_path):
    _install_broker(monkeypatch, tmp_path)
    client = JudgeClient(_judgment())
    sha = "a" * 40

    report = triage.investigate(
        issue=_issue(),
        repository="mlflow/mlflow",
        event_sha=sha,
        checkout_sha=sha,
        repo_root=tmp_path,
        client=client,
    )

    assert report["proposed_outcome"] == "ready"
    assert report["execution"]["stdout_excerpt"] == "trusted stdout\n"
    assert report["execution"]["executed_sha"] == sha
    assert len(client.calls) == 1
    assert client.calls[0]["output_schema"] == triage.JUDGMENT_SCHEMA


@pytest.mark.parametrize(
    "judgment",
    [
        _judgment(issue_kind="feature_request"),
        _judgment(surface="unsupported"),
        _judgment(
            fidelity={
                "verdict": "manufactured",
                "confidence": 0.99,
                "failure_origin": "trivial_assertion",
            }
        ),
        _judgment(
            fidelity={
                "verdict": "partial",
                "confidence": 0.9,
                "failure_origin": "reported_symptom",
            }
        ),
        _judgment(
            fidelity={
                "verdict": "faithful",
                "confidence": 0.9,
                "failure_origin": "unrelated_exception",
            }
        ),
        _judgment(proposed_fix={"scope": "broad", "confidence": 0.95}),
        _judgment(safety_uncertainty=True),
    ],
)
def test_uncertain_unsupported_or_broad_judgments_fail_closed(monkeypatch, tmp_path, judgment):
    _install_broker(monkeypatch, tmp_path)

    report = triage.investigate(
        issue=_issue(),
        repository="mlflow/mlflow",
        event_sha="a" * 40,
        checkout_sha="a" * 40,
        repo_root=tmp_path,
        client=JudgeClient(judgment),
    )

    assert report["proposed_outcome"] == "requires-further-triage"


def test_exact_enhancement_label_short_circuits_all_models_and_execution():
    class UnusedClient:
        def __getattribute__(self, name):
            raise AssertionError(f"client unexpectedly accessed: {name}")

    report = triage.investigate(
        issue=_issue([{"name": "enhancement"}]),
        repository="mlflow/mlflow",
        event_sha="a" * 40,
        checkout_sha="a" * 40,
        repo_root=Path("unused"),
        client=UnusedClient(),
    )

    assert report["issue_kind"] == "feature_request"
    assert report["proposed_outcome"] == "requires-further-triage"
    assert report["execution"]["executed_sha"] is None


def test_malformed_judgment_fails_at_sanitized_stage(monkeypatch, tmp_path):
    _install_broker(monkeypatch, tmp_path)

    with pytest.raises(triage.TriageStageError, match="judgment") as exc_info:
        triage.investigate(
            issue=_issue(),
            repository="mlflow/mlflow",
            event_sha="a" * 40,
            checkout_sha="a" * 40,
            repo_root=tmp_path,
            client=JudgeClient({"unexpected": "secret text"}),
        )

    error = exc_info.value
    report = triage._failure_report(
        repository="mlflow/mlflow",
        issue_number=123,
        event_sha="a" * 40,
        checkout_sha="a" * 40,
        stage=error.stage,
        error_class=error.error_class,
    )
    assert report["proposed_outcome"] == "requires-further-triage"
    assert report["failure"] == {"stage": "judgment", "error_class": "ValueError"}
    assert "secret" not in json.dumps(report)


def test_container_timeout_skips_judge_and_reports_only_failure_class(monkeypatch, tmp_path):
    _install_broker(monkeypatch, tmp_path, failure="timeout")
    client = JudgeClient(_judgment())

    with pytest.raises(triage.TriageStageError, match="container") as exc_info:
        triage.investigate(
            issue=_issue(),
            repository="mlflow/mlflow",
            event_sha="a" * 40,
            checkout_sha="a" * 40,
            repo_root=tmp_path,
            client=client,
        )

    assert exc_info.value.error_class == "ContainerTimeout"
    assert client.calls == []
