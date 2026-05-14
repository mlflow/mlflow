import pytest

from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    AssistantMessageAnchor,
    DedupVerdict,
    JobResponse,
    JobStatus,
    JudgeSpec,
    PersonaSpec,
    RunSummary,
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
    with pytest.raises(ValueError, match="strategy"):
        TestSpec(strategy="invalid", rationale_summary="x", assertion=AssertionSpec())


def test_assistant_message_anchor_required_fields():
    with pytest.raises(ValueError, match="Field required"):
        AssistantMessageAnchor()
    AssistantMessageAnchor(
        message_id="msg-789",
        start=120,
        end=178,
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


def test_verdict_passed_has_no_reasons():
    verdict = Verdict(test_case_id="tc-001", passed=True)
    assert verdict.passed
    assert verdict.reasons == ()
    assert verdict.judge_rationale is None


def test_verdict_failed_carries_reasons():
    verdict = Verdict(
        test_case_id="tc-001",
        passed=False,
        reasons=("must_call_tool: search_docs not invoked",),
    )
    assert not verdict.passed
    assert verdict.reasons == ("must_call_tool: search_docs not invoked",)


def test_run_summary_aggregates_counts():
    summary = RunSummary(
        run_id="run-xyz",
        pass_count=8,
        fail_count=1,
        error_count=0,
        duration_ms=12000,
    )
    assert summary.pass_count + summary.fail_count + summary.error_count == 9


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


@pytest.mark.parametrize(
    "kind",
    ["timeout", "missing_binary", "not_ready", "schema_validation", "other"],
)
def test_job_response_failure_kind_enum_values(kind: str):
    resp = JobResponse(job_id="j", status=JobStatus.FAILED, failure_kind=kind)
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
