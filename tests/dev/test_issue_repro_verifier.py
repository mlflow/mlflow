import copy
import hashlib
import json

import pytest

from dev.issue_repro_verifier import (
    MAX_MODEL_OUTPUT,
    MAX_REASON,
    InvalidVerification,
    build_verifier_state,
    parse_verifier_output,
    verify_reproduction,
)


def _case(
    *,
    claim="Calling mlflow.example() raises ValueError: invalid model URI.",
    source="import mlflow\nmlflow.example()\n",
    stdout="",
    stderr="ValueError: invalid model URI\n",
    exit_status=1,
    failure=None,
):
    encoded = source.encode()
    symptom = {
        "claim": claim,
        "verdict": "reproduced",
        "reproduction_steps": ["Run scratch/reproduce.py."],
        "stdout": stdout,
        "stderr": stderr,
        "exit_status": exit_status,
        "duration_seconds": 0.25,
        "artifact": {
            "relative_path": "scratch/reproduce.py",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        },
    }
    evidence = {
        "symptom_index": 0,
        "reproduction_source": source,
        "stdout": stdout,
        "stderr": stderr,
        "exit_status": exit_status,
        "duration_seconds": 0.25,
        "failure": failure,
    }
    return {"symptoms": [symptom]}, [evidence]


def _judgment(verdict, confidence=0.95, reason="The raw output faithfully matches the claim."):
    return {
        "symptoms": [
            {
                "symptom_index": 0,
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
            }
        ]
    }


class Client:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


@pytest.mark.parametrize(
    ("handoff", "evidence"),
    [
        _case(),
        _case(
            claim="mlflow.example() returns RESULT=0 instead of RESULT=42.",
            source="import mlflow\nprint(f'RESULT={mlflow.example()}')\n",
            stdout="RESULT=0\n",
            stderr="",
            exit_status=0,
        ),
    ],
)
def test_accepts_only_high_confidence_matching_exception_or_result(handoff, evidence):
    result = verify_reproduction(
        client=Client(_judgment("reproduced")),
        validated_handoff=handoff,
        raw_broker_evidence=evidence,
    )

    assert result[0]["verdict"] == "reproduced"
    assert result[0]["faithful_reproduction"] is True
    assert result[0]["confidence"] == 0.95


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        (
            _case(
                source="assert False, 'forced'\n",
                stderr="AssertionError: forced\n",
            ),
            "The scratch script contains only a trivial failing assertion.",
        ),
        (
            _case(
                source="import mlflow\nprint(mlflow.__version__)\n",
                stderr="KeyError: unrelated setup key\n",
            ),
            "The observed KeyError does not match the reported ValueError.",
        ),
        (
            _case(
                source="raise ValueError('invalid model URI')\n",
                stderr="ValueError: invalid model URI\n",
            ),
            "The scratch script creates the reported exception itself.",
        ),
        (
            _case(
                source="print('ValueError: invalid model URI')\n",
                stdout="ValueError: invalid model URI\n",
                stderr="",
                exit_status=0,
            ),
            "The script merely prints a fabricated failure summary.",
        ),
    ],
)
def test_trivial_unrelated_self_induced_and_fabricated_failures_are_inconclusive(case, reason):
    handoff, evidence = case

    result = verify_reproduction(
        client=Client(_judgment("inconclusive", reason=reason)),
        validated_handoff=handoff,
        raw_broker_evidence=evidence,
    )

    assert result[0]["verdict"] == "inconclusive"
    assert result[0]["faithful_reproduction"] is False


def test_partial_symptom_match_does_not_turn_every_claim_into_reproduced():
    handoff, evidence = _case()
    second_handoff, second_evidence = _case(
        claim="The same call also corrupts the tracking database.",
        stdout="",
        stderr="ValueError: invalid model URI\n",
    )
    handoff["symptoms"].append(second_handoff["symptoms"][0])
    second_evidence[0]["symptom_index"] = 1
    evidence.extend(second_evidence)
    output = {
        "symptoms": [
            {
                "symptom_index": 0,
                "verdict": "reproduced",
                "confidence": 0.96,
                "reason": "The exception matches the first claim.",
            },
            {
                "symptom_index": 1,
                "verdict": "inconclusive",
                "confidence": 0.99,
                "reason": "The raw evidence says nothing about database corruption.",
            },
        ]
    }

    result = verify_reproduction(
        client=Client(output),
        validated_handoff=handoff,
        raw_broker_evidence=evidence,
    )

    assert [item["faithful_reproduction"] for item in result] == [True, False]
    assert [item["verdict"] for item in result] == ["reproduced", "inconclusive"]


def test_missing_raw_output_is_rejected_before_calling_verifier():
    handoff, evidence = _case(stdout="", stderr="", exit_status=1)
    client = Client(_judgment("reproduced"))

    with pytest.raises(InvalidVerification, match="missing raw output"):
        verify_reproduction(
            client=client,
            validated_handoff=handoff,
            raw_broker_evidence=evidence,
        )

    assert client.calls == []


def test_exit_status_alone_cannot_be_raw_reproduction_evidence():
    handoff, evidence = _case(stdout="", stderr="", exit_status=23)

    with pytest.raises(InvalidVerification, match="missing raw output"):
        build_verifier_state(handoff, evidence)


def test_broker_failure_cannot_be_accepted_as_reproduction():
    handoff, evidence = _case(
        source="while True:\n    pass\n",
        stdout="process timed out\n",
        stderr="",
        exit_status=None,
        failure="timeout",
    )

    result = verify_reproduction(
        client=Client(_judgment("reproduced", confidence=0.99)),
        validated_handoff=handoff,
        raw_broker_evidence=evidence,
    )

    assert result[0]["verdict"] == "inconclusive"
    assert result[0]["faithful_reproduction"] is False


def test_verifier_receives_only_bound_evidence_and_no_tool_interface():
    handoff, evidence = _case()
    handoff["symptoms"][0]["verdict"] = "not_reproduced"
    client = Client(_judgment("reproduced"))

    verify_reproduction(
        client=client,
        validated_handoff=handoff,
        raw_broker_evidence=evidence,
    )

    assert set(client.calls[0]) == {"messages", "output_schema"}
    assert "tools" not in client.calls[0]
    payload = json.loads(client.calls[0]["messages"][1]["content"].split("\n", 1)[1])
    assert payload["symptoms"][0]["raw_stderr"] == evidence[0]["stderr"]
    assert "verdict" not in payload["symptoms"][0]


def test_low_confidence_reproduction_is_normalized_to_inconclusive():
    result = parse_verifier_output(_judgment("reproduced", confidence=0.79), symptom_count=1)

    assert result[0]["verdict"] == "inconclusive"
    assert result[0]["faithful_reproduction"] is False
    assert result[0]["confidence"] == 0.79


def test_high_confidence_not_reproduced_is_not_a_faithful_reproduction():
    result = parse_verifier_output(_judgment("not_reproduced"), symptom_count=1)

    assert result[0]["verdict"] == "not_reproduced"
    assert result[0]["faithful_reproduction"] is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=True), "fields"),
        (lambda value: value["symptoms"][0].update(extra=True), "fields"),
        (lambda value: value["symptoms"][0].update(verdict="yes"), "verdict"),
        (lambda value: value["symptoms"][0].update(confidence=True), "confidence"),
        (lambda value: value["symptoms"][0].update(confidence=float("nan")), "confidence"),
        (lambda value: value["symptoms"][0].update(confidence=1.01), "confidence"),
        (lambda value: value["symptoms"][0].update(reason="x" * (MAX_REASON + 1)), "reason"),
        (lambda value: value["symptoms"][0].update(symptom_index=1), "index"),
        (lambda value: value["symptoms"].append(copy.deepcopy(value["symptoms"][0])), "count"),
    ],
)
def test_verifier_output_schema_and_bounds_fail_closed(mutate, match):
    output = _judgment("reproduced")
    mutate(output)

    with pytest.raises(InvalidVerification, match=match):
        parse_verifier_output(output, symptom_count=1)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json", "malformed"),
        ('{"symptoms": [], "symptoms": []}', "duplicate"),
        (b"\xff", "malformed"),
        ("x" * (MAX_MODEL_OUTPUT + 1), "oversized"),
    ],
)
def test_verifier_json_parsing_fails_closed(payload, match):
    with pytest.raises(InvalidVerification, match=match):
        parse_verifier_output(payload, symptom_count=1)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda handoff, evidence: evidence[0].update(stdout="different"), "does not match"),
        (lambda handoff, evidence: evidence[0].update(failure="network"), "broker failure"),
        (lambda handoff, evidence: evidence[0].update(symptom_index=True), "symptom index"),
        (lambda handoff, evidence: evidence[0].update(extra="x"), "fields"),
        (
            lambda handoff, evidence: handoff["symptoms"][0]["artifact"].update(sha256="0" * 64),
            "does not match source",
        ),
    ],
)
def test_raw_evidence_schema_and_binding_fail_closed(mutate, match):
    handoff, evidence = _case()
    mutate(handoff, evidence)

    with pytest.raises(InvalidVerification, match=match):
        build_verifier_state(handoff, evidence)
