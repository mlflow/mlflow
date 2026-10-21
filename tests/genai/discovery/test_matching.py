import json
from unittest import mock

import pytest

import mlflow
from mlflow.entities.issue import Issue, IssueStatus
from mlflow.genai.discovery.constants import SAME_ISSUE_RULE
from mlflow.genai.discovery.entities import _IdentifiedIssue
from mlflow.genai.discovery.matching import (
    MATCHABLE_STATUSES,
    CandidateReadError,
    MatchOutcome,
    match_findings_to_existing_issues,
    read_match_candidates,
)


def _finding(name, description="desc", root_cause="cause"):
    return _IdentifiedIssue(
        name=name,
        description=description,
        root_cause=root_cause,
        severity="medium",
        categories=["correctness"],
        category_rationale="correctness: wrong output",
        analysis_indices=[],
        label_indices=[],
        example_indices=[],
    )


def _existing(issue_id, name, description="stored desc", root_causes=None):
    return Issue(
        issue_id=issue_id,
        experiment_id="1",
        name=name,
        description=description,
        status=IssueStatus.PENDING,
        severity="high",
        root_causes=root_causes or ["stored cause"],
        categories=["correctness"],
        created_timestamp=1,
        last_updated_timestamp=1,
    )


def _matcher(payload):
    response = mock.MagicMock()
    response.choices[0].message.content = json.dumps(payload)
    return mock.patch("mlflow.genai.discovery.pipeline._call_llm", return_value=response)


def test_the_same_issue_test_is_shared_with_within_run_dedup():
    from mlflow.genai.discovery.constants import (
        CROSS_RUN_MATCH_PROMPT_TEMPLATE,
        DEDUP_ISSUES_PROMPT_TEMPLATE,
    )

    # One rule, not two copies: a finding merged into a sibling this run must be the same
    # finding that matches a stored issue next run.
    assert SAME_ISSUE_RULE in DEDUP_ISSUES_PROMPT_TEMPLATE
    assert SAME_ISSUE_RULE in CROSS_RUN_MATCH_PROMPT_TEMPLATE


def test_a_finding_already_reported_is_matched_to_its_issue():
    findings = [_finding("Refund falsely confirmed")]
    candidates = [_existing("iss-known", "Refund completion falsely confirmed")]

    with _matcher({"matches": [{"finding_index": 0, "issue_id": "iss-known"}]}):
        outcome = match_findings_to_existing_issues(findings, candidates)

    assert outcome.issue_id_for(0) == "iss-known"
    assert outcome.already_reported == 1
    assert outcome.candidates_considered == 1


def test_a_genuinely_new_finding_is_left_unmatched():
    findings = [_finding("Language ignored")]
    candidates = [_existing("iss-known", "Refund falsely confirmed")]

    with _matcher({"matches": [{"finding_index": 0, "issue_id": None}]}):
        outcome = match_findings_to_existing_issues(findings, candidates)

    assert outcome.issue_id_for(0) is None
    assert outcome.already_reported == 0


def test_a_match_to_an_issue_that_was_never_offered_is_ignored():
    # A stale or foreign ID would attach this run's evidence to an issue in another experiment,
    # or to nothing at all.
    findings = [_finding("Refund falsely confirmed")]
    candidates = [_existing("iss-known", "Refund falsely confirmed")]

    with _matcher({"matches": [{"finding_index": 0, "issue_id": "iss-elsewhere"}]}):
        outcome = match_findings_to_existing_issues(findings, candidates)

    assert outcome.matched == {}


def test_a_match_for_a_finding_that_does_not_exist_is_ignored():
    findings = [_finding("Refund falsely confirmed")]
    candidates = [_existing("iss-known", "Refund falsely confirmed")]

    with _matcher({"matches": [{"finding_index": 7, "issue_id": "iss-known"}]}):
        outcome = match_findings_to_existing_issues(findings, candidates)

    assert outcome.matched == {}


def test_a_matcher_failure_reports_findings_as_new_rather_than_guessing():
    findings = [_finding("Refund falsely confirmed")]
    candidates = [_existing("iss-known", "Refund falsely confirmed")]

    with mock.patch("mlflow.genai.discovery.pipeline._call_llm", side_effect=RuntimeError("down")):
        outcome = match_findings_to_existing_issues(findings, candidates)

    # Duplicating an issue is recoverable; folding a finding into the wrong one hides it.
    assert outcome.matched == {}


def test_nothing_reported_yet_needs_no_matcher_call():
    with mock.patch("mlflow.genai.discovery.pipeline._call_llm") as called:
        outcome = match_findings_to_existing_issues([_finding("a")], [])

    called.assert_not_called()
    assert outcome.matched == {}


def test_a_failed_candidate_read_raises_rather_than_looking_empty():
    # Publishing against an unread list would file a duplicate of every known issue.
    with mock.patch(
        "mlflow.tracing.client.TracingClient._search_issues", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(CandidateReadError, match="already reported"):
            read_match_candidates("1")


def test_only_pending_and_rejected_issues_are_offered():
    # A resolved issue is not matchable: the same problem after a fix is a regression, which is
    # new information and gets its own issue.
    assert set(MATCHABLE_STATUSES) == {IssueStatus.PENDING.value, IssueStatus.REJECTED.value}
    assert IssueStatus.RESOLVED.value not in MATCHABLE_STATUSES

    with mock.patch(
        "mlflow.tracing.client.TracingClient._search_issues", return_value=[]
    ) as searched:
        read_match_candidates("1")

    assert searched.call_args.kwargs["statuses"] == list(MATCHABLE_STATUSES)


def test_the_candidate_list_is_bounded():
    from mlflow.genai.discovery.constants import MAX_MATCH_CANDIDATES

    with mock.patch(
        "mlflow.tracing.client.TracingClient._search_issues", return_value=[]
    ) as searched:
        read_match_candidates("1")

    assert searched.call_args.kwargs["max_results"] == MAX_MATCH_CANDIDATES


# ---- publication ----


def test_a_repeat_run_refreshes_the_issue_instead_of_filing_another():
    from mlflow.genai.discovery.pipeline import _build_issues

    exp_id = mlflow.set_experiment("b7-repeat").experiment_id
    from mlflow.tracing.client import TracingClient

    client = TracingClient()
    first = client._create_issue(
        experiment_id=exp_id,
        name="Generic policy answers omit details",
        description="The agent answers with a generic policy line.",
        severity="medium",
        categories=["relevance"],
        root_causes=["answer step ignores the retrieved document"],
    )

    finding = _finding("Generic policy answers omit details")
    outcome = MatchOutcome(matched={0: first.issue_id}, candidates_considered=1)
    issues, issue_trace_ids = _build_issues([finding], [], exp_id, "run-2", outcome)

    assert [issue.issue_id for issue in issues] == [first.issue_id]
    assert issue_trace_ids[first.issue_id] == []
    # One row, not two: the tab shows the problem once however often detection runs.
    stored = client._search_issues(experiment_id=exp_id)
    assert len(list(stored)) == 1


def test_a_refreshed_issue_keeps_the_triage_and_wording_a_user_gave_it():
    from mlflow.genai.discovery.pipeline import _build_issues
    from mlflow.tracing.client import TracingClient

    exp_id = mlflow.set_experiment("b7-triage").experiment_id
    client = TracingClient()
    existing = client._create_issue(
        experiment_id=exp_id,
        name="Operator's own title",
        description="Operator's own description.",
        severity="low",
        categories=["relevance"],
    )
    client._update_issue(issue_id=existing.issue_id, status=IssueStatus.REJECTED)

    finding = _finding("Detector's wording", description="Detector's description.")
    _build_issues([finding], [], exp_id, "run-2", MatchOutcome(matched={0: existing.issue_id}))

    refreshed = client._get_issue(existing.issue_id)
    # Detection refreshes when the issue was last seen; it does not overturn a human decision
    # or overwrite the text a human wrote.
    assert refreshed.status == IssueStatus.REJECTED
    assert refreshed.name == "Operator's own title"
    assert refreshed.description == "Operator's own description."
    assert refreshed.last_seen_timestamp is not None


def test_an_unmatched_finding_is_still_filed_as_a_new_issue():
    from mlflow.genai.discovery.pipeline import _build_issues
    from mlflow.tracing.client import TracingClient

    exp_id = mlflow.set_experiment("b7-new").experiment_id
    with mlflow.start_run() as run:
        issues, _ = _build_issues(
            [_finding("Brand new problem")], [], exp_id, run.info.run_id, MatchOutcome()
        )

    assert len(issues) == 1
    assert issues[0].source_run_id == run.info.run_id
    assert len(list(TracingClient()._search_issues(experiment_id=exp_id))) == 1


def test_a_refresh_that_fails_falls_back_to_filing_the_finding():
    from mlflow.genai.discovery.pipeline import _build_issues
    from mlflow.tracing.client import TracingClient

    exp_id = mlflow.set_experiment("b7-refresh-fails").experiment_id
    outcome = MatchOutcome(matched={0: "iss-gone"})
    with (
        mlflow.start_run() as run,
        mock.patch.object(TracingClient, "_update_issue", side_effect=RuntimeError("vanished")),
    ):
        issues, _ = _build_issues(
            [_finding("Still worth reporting")], [], exp_id, run.info.run_id, outcome
        )

    # Losing the match costs a duplicate; losing the finding costs the user the problem.
    assert len(issues) == 1
    assert issues[0].name == "Still worth reporting"
