import pytest
from pydantic import TypeAdapter, ValidationError

from mlflow.agent_playground.test_cases.entities import (
    _SIMULATOR_RESERVED_CONTEXT_KEYS,
    AssertionExpectations,
    AssistantMessageAnchor,
    DedupVerdict,
    Expectations,
    JudgeExpectations,
    PersonaSpec,
    Verdict,
)

_EXPECTATIONS_ADAPTER = TypeAdapter(Expectations)


# ---------------------------------------------------------------------------
# AssertionExpectations
# ---------------------------------------------------------------------------


def test_assertion_expectations_defaults_to_empty_lists():
    spec = AssertionExpectations()
    assert spec.kind == "assertion"
    assert spec.must_contain == []
    assert spec.must_not_contain == []
    assert spec.must_call_tool == []
    assert spec.must_not_call_tool == []


@pytest.mark.parametrize(
    ("field_name", "blank_value"),
    [
        ("must_contain", ""),
        ("must_contain", "   "),
        ("must_not_contain", ""),
        ("must_call_tool", "\t"),
        ("must_not_call_tool", ""),
    ],
)
def test_assertion_expectations_rejects_blank_clauses(field_name: str, blank_value: str):
    with pytest.raises(ValueError, match="must be non-empty"):
        AssertionExpectations(**{field_name: [blank_value]})


# ---------------------------------------------------------------------------
# JudgeExpectations
# ---------------------------------------------------------------------------


def test_judge_expectations_requires_criteria():
    with pytest.raises(ValueError, match="criteria"):
        JudgeExpectations()
    JudgeExpectations(criteria="must cite docs")


def test_judge_expectations_kind_is_judge():
    spec = JudgeExpectations(criteria="response is friendly")
    assert spec.kind == "judge"
    assert spec.expected_response is None


# ---------------------------------------------------------------------------
# Expectations discriminated union
# ---------------------------------------------------------------------------


def test_expectations_dispatches_assertion_kind():
    obj = _EXPECTATIONS_ADAPTER.validate_python(
        {"kind": "assertion", "must_contain": ["docs"]},
    )
    assert isinstance(obj, AssertionExpectations)
    assert obj.must_contain == ["docs"]


def test_expectations_dispatches_judge_kind():
    obj = _EXPECTATIONS_ADAPTER.validate_python(
        {"kind": "judge", "criteria": "be helpful"},
    )
    assert isinstance(obj, JudgeExpectations)
    assert obj.criteria == "be helpful"


def test_expectations_rejects_unknown_kind():
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        _EXPECTATIONS_ADAPTER.validate_python({"kind": "magic", "criteria": "x"})


def test_expectations_requires_kind_discriminator():
    with pytest.raises(ValidationError, match="Unable to extract tag using discriminator"):
        _EXPECTATIONS_ADAPTER.validate_python({"must_contain": ["docs"]})


# ---------------------------------------------------------------------------
# PersonaSpec
# ---------------------------------------------------------------------------


def test_persona_spec_requires_goal():
    with pytest.raises(ValueError, match="goal"):
        PersonaSpec()
    PersonaSpec(goal="understand logging")


def test_persona_spec_model_dump_matches_simulator_dict_shape():
    spec = PersonaSpec(
        goal="understand logging conventions",
        persona="an intermediate Python developer",
        simulation_guidelines=["ask follow-up questions"],
        context={"user_id": "123"},
    )
    dumped = spec.model_dump(exclude_none=True)
    assert dumped == {
        "goal": "understand logging conventions",
        "persona": "an intermediate Python developer",
        "simulation_guidelines": ["ask follow-up questions"],
        "context": {"user_id": "123"},
    }


def test_persona_spec_omits_none_fields_on_dump():
    # Sparse persona should dump cleanly without nulls so the simulator
    # treats absent fields as defaults.
    spec = PersonaSpec(goal="learn how logging works")
    assert spec.model_dump(exclude_none=True) == {"goal": "learn how logging works"}


@pytest.mark.parametrize(
    "reserved_key",
    ["input", "messages", "mlflow_session_id"],
)
def test_persona_spec_rejects_simulator_reserved_context_keys(reserved_key: str):
    with pytest.raises(ValueError, match="simulator-reserved"):
        PersonaSpec(goal="g", context={reserved_key: "x"})


def test_persona_spec_accepts_non_reserved_context_keys():
    spec = PersonaSpec(goal="g", context={"user_id": "123", "tenant": "acme"})
    assert spec.context == {"user_id": "123", "tenant": "acme"}


def test_persona_spec_reserved_keys_stay_in_lockstep_with_simulator():
    # Drift guard: if the simulator adds a fourth reserved key,
    # ``_SIMULATOR_RESERVED_CONTEXT_KEYS`` must be updated in lockstep
    # or PersonaSpec.context will accept invalid input that the
    # simulator rejects at runtime.
    from mlflow.genai.simulators.simulator import _RESERVED_CONTEXT_KEYS

    assert _SIMULATOR_RESERVED_CONTEXT_KEYS == _RESERVED_CONTEXT_KEYS


def test_persona_spec_fields_are_subset_of_simulator_expected_keys():
    # Drift guard: PersonaSpec.model_dump(exclude_none=True) is handed
    # straight to ConversationSimulator, which warns on unexpected
    # keys. Catching subset drift in CI is cheaper than the warning
    # showing up at run time.
    from mlflow.genai.simulators.simulator import _EXPECTED_TEST_CASE_KEYS

    assert set(PersonaSpec.model_fields) <= _EXPECTED_TEST_CASE_KEYS


# ---------------------------------------------------------------------------
# AssistantMessageAnchor / TraceAnchor
# ---------------------------------------------------------------------------


def test_assistant_message_anchor_required_fields():
    with pytest.raises(ValueError, match="Field required"):
        AssistantMessageAnchor()
    AssistantMessageAnchor(
        message_id="msg-789",
        start=120,
        end=140,
        selected_text="use INFO for general",
        prefix="answer: ",
        suffix=" hope that",
    )


def test_assistant_message_anchor_kind_defaults_to_assistant_message():
    anchor = AssistantMessageAnchor(
        message_id="msg-1",
        start=0,
        end=5,
        selected_text="hello",
        prefix="",
        suffix="",
    )
    assert anchor.kind == "assistant_message"


def test_assistant_message_anchor_trace_id_optional():
    anchor = AssistantMessageAnchor(
        message_id="msg-1",
        start=0,
        end=5,
        selected_text="hello",
        prefix="",
        suffix="",
    )
    assert anchor.trace_id is None


def test_assistant_message_anchor_rejects_negative_start():
    with pytest.raises(ValueError, match="start must be non-negative"):
        AssistantMessageAnchor(
            message_id="m", start=-1, end=4, selected_text="abcd", prefix="", suffix=""
        )


def test_assistant_message_anchor_rejects_end_before_start():
    with pytest.raises(ValueError, match="must be >= start"):
        AssistantMessageAnchor(
            message_id="m", start=10, end=4, selected_text="", prefix="", suffix=""
        )


def test_assistant_message_anchor_rejects_mismatched_selected_text_length():
    with pytest.raises(ValueError, match="must equal end - start"):
        AssistantMessageAnchor(
            message_id="m", start=0, end=5, selected_text="hi", prefix="", suffix=""
        )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_verdict_pass_has_no_reasons():
    verdict = Verdict(test_case_id="tc-001", outcome="pass")
    assert verdict.outcome == "pass"
    assert verdict.reasons == ()
    assert verdict.judge_rationale is None


def test_verdict_fail_carries_reasons():
    verdict = Verdict(
        test_case_id="tc-001",
        outcome="fail",
        reasons=("must_call_tool: search_docs not invoked",),
    )
    assert verdict.outcome == "fail"
    assert verdict.reasons == ("must_call_tool: search_docs not invoked",)


def test_verdict_error_carries_reasons():
    verdict = Verdict(
        test_case_id="tc-001",
        outcome="error",
        reasons=("agent crashed mid-turn",),
    )
    assert verdict.outcome == "error"
    assert verdict.reasons == ("agent crashed mid-turn",)


def test_verdict_pass_with_reasons_rejected():
    with pytest.raises(ValueError, match="must not carry reasons"):
        Verdict(test_case_id="tc-001", outcome="pass", reasons=("nope",))


def test_verdict_invalid_outcome_rejected():
    with pytest.raises(ValueError, match="outcome"):
        Verdict(test_case_id="tc-001", outcome="maybe")


# ---------------------------------------------------------------------------
# DedupVerdict
# ---------------------------------------------------------------------------


def test_dedup_verdict_unique_shape():
    verdict = DedupVerdict(is_duplicate=False, reason="no match found")
    assert not verdict.is_duplicate
    assert verdict.existing_test_case_id is None


def test_dedup_verdict_duplicate_shape():
    verdict = DedupVerdict(
        is_duplicate=True,
        existing_test_case_id="tc-001",
        reason="same complaint about log levels",
    )
    assert verdict.is_duplicate
    assert verdict.existing_test_case_id == "tc-001"


def test_dedup_verdict_duplicate_without_id_rejected():
    with pytest.raises(ValueError, match="requires existing_test_case_id"):
        DedupVerdict(is_duplicate=True, reason="duplicate but unspecified")


def test_dedup_verdict_unique_with_id_rejected():
    with pytest.raises(ValueError, match="must not carry existing_test_case_id"):
        DedupVerdict(
            is_duplicate=False,
            existing_test_case_id="tc-001",
            reason="actually unique",
        )
