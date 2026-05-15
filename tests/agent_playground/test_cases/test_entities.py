import pytest

from mlflow.agent_playground.test_cases.entities import (
    _SIMULATOR_RESERVED_CONTEXT_KEYS,
    AssertionSpec,
    AssistantMessageAnchor,
    DedupVerdict,
    JobResponse,
    JobStatus,
    JudgeSpec,
    PersonaSpec,
    TestSpec,
    Verdict,
)


def test_assertion_spec_defaults_to_empty_lists():
    spec = AssertionSpec()
    assert spec.must_contain == []
    assert spec.must_not_contain == []
    assert spec.must_call_tool == []
    assert spec.must_not_call_tool == []


def test_judge_spec_requires_criteria():
    with pytest.raises(ValueError, match="criteria"):
        JudgeSpec()
    JudgeSpec(criteria="must cite docs")


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


def test_test_spec_assertion_strategy_requires_assertion_payload():
    with pytest.raises(ValueError, match="assertion payload"):
        TestSpec(strategy="assertion", rationale_summary="agent must cite docs")
    TestSpec(
        strategy="assertion",
        rationale_summary="agent must cite docs",
        assertion=AssertionSpec(must_contain=["docs"]),
    )


def test_test_spec_judge_strategy_requires_judge_payload():
    with pytest.raises(ValueError, match="judge payload"):
        TestSpec(strategy="judge", rationale_summary="agent must sound friendlier")
    TestSpec(
        strategy="judge",
        rationale_summary="agent must sound friendlier",
        judge=JudgeSpec(criteria="response is friendly"),
    )


def test_test_spec_default_max_turns_is_5():
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="x",
        assertion=AssertionSpec(),
    )
    assert spec.max_turns == 5


def test_test_spec_max_turns_overridable_per_case():
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="x",
        max_turns=10,
        assertion=AssertionSpec(),
    )
    assert spec.max_turns == 10


def test_test_spec_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="Input should be 'assertion' or 'judge'"):
        TestSpec(strategy="invalid", rationale_summary="x", assertion=AssertionSpec())


def test_test_spec_assertion_strategy_rejects_orphan_judge_payload():
    with pytest.raises(ValueError, match="must not carry a judge payload"):
        TestSpec(
            strategy="assertion",
            rationale_summary="x",
            assertion=AssertionSpec(must_contain=["docs"]),
            judge=JudgeSpec(criteria="c"),
        )


def test_test_spec_judge_strategy_rejects_orphan_assertion_payload():
    with pytest.raises(ValueError, match="must not carry an assertion payload"):
        TestSpec(
            strategy="judge",
            rationale_summary="x",
            assertion=AssertionSpec(must_contain=["docs"]),
            judge=JudgeSpec(criteria="c"),
        )


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


def test_job_status_enum_values():
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.SUCCEEDED.value == "succeeded"
    assert JobStatus.FAILED.value == "failed"


def test_job_response_pending_shape():
    resp = JobResponse(job_id="job-1", status=JobStatus.PENDING)
    assert resp.test_case_id is None
    assert not resp.deduped
    assert resp.failure_kind is None


def test_job_response_succeeded_with_dedup():
    resp = JobResponse(
        job_id="job-1",
        status=JobStatus.SUCCEEDED,
        test_case_id="tc-001",
        deduped=True,
    )
    assert resp.test_case_id == "tc-001"
    assert resp.deduped


def test_job_response_failed_with_kind():
    resp = JobResponse(
        job_id="job-1",
        status=JobStatus.FAILED,
        failure_kind="missing_binary",
        failure_reason="claude CLI not on PATH",
    )
    assert resp.failure_kind == "missing_binary"


def test_job_response_failed_without_kind_rejected():
    with pytest.raises(ValueError, match="failure_kind"):
        JobResponse(job_id="j", status=JobStatus.FAILED, failure_reason="x")


def test_job_response_failed_without_reason_rejected():
    with pytest.raises(ValueError, match="failure_reason"):
        JobResponse(job_id="j", status=JobStatus.FAILED, failure_kind="other")


@pytest.mark.parametrize(
    "non_failed_status",
    [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.CANCELLED],
)
def test_job_response_non_failed_must_not_carry_failure_kind(non_failed_status: JobStatus):
    with pytest.raises(ValueError, match="must not carry failure_kind"):
        JobResponse(job_id="j", status=non_failed_status, failure_kind="other")


@pytest.mark.parametrize(
    "non_failed_status",
    [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED, JobStatus.CANCELLED],
)
def test_job_response_non_failed_must_not_carry_failure_reason(non_failed_status: JobStatus):
    with pytest.raises(ValueError, match="must not carry failure_reason"):
        JobResponse(job_id="j", status=non_failed_status, failure_reason="x")


@pytest.mark.parametrize(
    "kind",
    ["timeout", "missing_binary", "not_ready", "schema_validation", "other"],
)
def test_job_response_failure_kind_enum_values(kind: str):
    resp = JobResponse(
        job_id="j",
        status=JobStatus.FAILED,
        failure_kind=kind,
        failure_reason="ctx",
    )
    assert resp.failure_kind == kind


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
