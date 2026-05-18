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
    ids=[
        "must_contain-empty",
        "must_contain-whitespace",
        "must_not_contain-empty",
        "must_call_tool-tab",
        "must_not_call_tool-empty",
    ],
)
def test_assertion_expectations_rejects_blank_clauses(field_name: str, blank_value: str):
    with pytest.raises(ValueError, match="must be non-empty"):
        AssertionExpectations(**{field_name: [blank_value]})


@pytest.mark.parametrize(
    ("field_a", "field_b"),
    [
        ("must_contain", "must_not_contain"),
        ("must_call_tool", "must_not_call_tool"),
    ],
)
def test_assertion_expectations_rejects_contradictory_clauses(field_a: str, field_b: str):
    with pytest.raises(ValueError, match="overlap"):
        AssertionExpectations(**{field_a: ["docs"], field_b: ["docs"]})


def test_assertion_expectations_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssertionExpectations(must_contains=["docs"])


def test_assertion_expectations_is_frozen():
    spec = AssertionExpectations(must_contain=["docs"])
    with pytest.raises(ValidationError, match="frozen"):
        spec.must_contain = ["nope"]


# ---------------------------------------------------------------------------
# JudgeExpectations
# ---------------------------------------------------------------------------


def test_judge_expectations_requires_instructions():
    with pytest.raises(ValidationError, match="instructions"):
        JudgeExpectations()


def test_judge_expectations_accepts_non_empty_instructions():
    spec = JudgeExpectations(instructions="must cite docs")
    assert spec.kind == "judge"
    assert spec.instructions == "must cite docs"
    assert spec.expected_response is None


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"], ids=["empty", "whitespace", "tab-newline"])
def test_judge_expectations_rejects_blank_instructions(blank: str):
    with pytest.raises(ValidationError, match="instructions"):
        JudgeExpectations(instructions=blank)


def test_judge_expectations_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JudgeExpectations(instructions="be helpful", criteria="redundant")


def test_judge_expectations_is_frozen():
    spec = JudgeExpectations(instructions="be helpful")
    with pytest.raises(ValidationError, match="frozen"):
        spec.instructions = "be cheeky"


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
        {"kind": "judge", "instructions": "be helpful"},
    )
    assert isinstance(obj, JudgeExpectations)
    assert obj.instructions == "be helpful"


def test_expectations_rejects_unknown_kind():
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        _EXPECTATIONS_ADAPTER.validate_python({"kind": "magic", "instructions": "x"})


def test_expectations_requires_kind_discriminator():
    with pytest.raises(ValidationError, match="Unable to extract tag using discriminator"):
        _EXPECTATIONS_ADAPTER.validate_python({"must_contain": ["docs"]})


# ---------------------------------------------------------------------------
# PersonaSpec
# ---------------------------------------------------------------------------


def test_persona_spec_requires_goal():
    with pytest.raises(ValidationError, match="goal"):
        PersonaSpec()


def test_persona_spec_accepts_minimal_goal_only():
    spec = PersonaSpec(goal="understand logging")
    assert spec.goal == "understand logging"
    assert spec.persona is None


@pytest.mark.parametrize("blank", ["", "   ", "\n"], ids=["empty", "whitespace", "newline"])
def test_persona_spec_rejects_blank_goal(blank: str):
    with pytest.raises(ValidationError, match="goal"):
        PersonaSpec(goal=blank)


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


def test_persona_spec_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PersonaSpec(goal="g", goals=["typo'd"])


def test_persona_spec_is_frozen():
    spec = PersonaSpec(goal="g")
    with pytest.raises(ValidationError, match="frozen"):
        spec.goal = "different"


def test_persona_spec_reserved_keys_stay_in_lockstep_with_simulator():
    from mlflow.genai.simulators.simulator import _RESERVED_CONTEXT_KEYS

    assert _SIMULATOR_RESERVED_CONTEXT_KEYS == _RESERVED_CONTEXT_KEYS


def test_persona_spec_fields_are_subset_of_simulator_expected_keys():
    from mlflow.genai.simulators.simulator import _EXPECTED_TEST_CASE_KEYS

    assert set(PersonaSpec.model_fields) <= _EXPECTED_TEST_CASE_KEYS


def test_persona_spec_simulation_guidelines_shape_matches_simulator():
    # Drift guard: the simulator's public ``SimulatorContext`` dataclass
    # accepts ``str | list[str] | None`` for simulation_guidelines.
    # PersonaSpec deliberately narrows to ``list[str] | None`` for UI
    # editability, and the runner normalizes a single string to a
    # one-item list before handoff. If the simulator ever drops
    # ``list[str]`` from its accepted shape, this drift guard fails
    # so the runner gets updated in lockstep.
    from mlflow.genai.simulators.simulator import SimulatorContext

    annotation_str = str(SimulatorContext.__annotations__["simulation_guidelines"])
    assert "list[str]" in annotation_str or "List[str]" in annotation_str, (
        f"simulator no longer accepts list[str] for simulation_guidelines "
        f"(got {annotation_str!r}); PersonaSpec narrowing is now incompatible"
    )


# ---------------------------------------------------------------------------
# AssistantMessageAnchor / TraceAnchor
# ---------------------------------------------------------------------------


def test_assistant_message_anchor_required_fields_missing():
    with pytest.raises(ValidationError, match="Field required"):
        AssistantMessageAnchor()


def test_assistant_message_anchor_accepts_well_formed_input():
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


def test_assistant_message_anchor_rejects_empty_message_id():
    with pytest.raises(ValidationError, match="message_id"):
        AssistantMessageAnchor(
            message_id="", start=0, end=5, selected_text="hello", prefix="", suffix=""
        )


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


def test_assistant_message_anchor_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AssistantMessageAnchor(
            message_id="m",
            start=0,
            end=5,
            selected_text="hello",
            prefix="",
            suffix="",
            unknown_field="oops",
        )


def test_assistant_message_anchor_is_frozen():
    anchor = AssistantMessageAnchor(
        message_id="m", start=0, end=5, selected_text="hello", prefix="", suffix=""
    )
    with pytest.raises(ValidationError, match="frozen"):
        anchor.start = 1


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


@pytest.mark.parametrize("outcome", ["fail", "error"])
def test_verdict_fail_or_error_requires_reasons(outcome: str):
    with pytest.raises(ValueError, match="must carry at least one reason"):
        Verdict(test_case_id="tc-001", outcome=outcome)


def test_verdict_invalid_outcome_rejected():
    with pytest.raises(ValueError, match="outcome"):
        Verdict(test_case_id="tc-001", outcome="maybe", reasons=("x",))


def test_verdict_rejects_empty_test_case_id():
    with pytest.raises(ValidationError, match="test_case_id"):
        Verdict(test_case_id="", outcome="pass")


def test_verdict_rejects_negative_duration_ms():
    with pytest.raises(ValidationError, match="duration_ms"):
        Verdict(test_case_id="tc-001", outcome="pass", duration_ms=-1)


def test_verdict_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Verdict(test_case_id="tc-001", outcome="pass", outcomes="typo'd")


def test_verdict_is_frozen():
    verdict = Verdict(test_case_id="tc-001", outcome="pass")
    with pytest.raises(ValidationError, match="frozen"):
        verdict.outcome = "fail"


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


def test_dedup_verdict_rejects_empty_reason():
    with pytest.raises(ValidationError, match="reason"):
        DedupVerdict(is_duplicate=False, reason="")


def test_dedup_verdict_rejects_empty_existing_test_case_id():
    with pytest.raises(ValidationError, match="existing_test_case_id"):
        DedupVerdict(is_duplicate=True, existing_test_case_id="", reason="x")


def test_dedup_verdict_rejects_unknown_field():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DedupVerdict(is_duplicate=False, reason="unique", extra_flag=True)


def test_dedup_verdict_is_frozen():
    verdict = DedupVerdict(is_duplicate=False, reason="unique")
    with pytest.raises(ValidationError, match="frozen"):
        verdict.reason = "changed mind"
