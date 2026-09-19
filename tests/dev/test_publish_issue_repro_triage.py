import pytest

from dev.publish_issue_repro_triage import (
    LABELS,
    MARKER,
    GitHubClient,
    build_publish_plan,
    validate_issue,
    validate_report,
)

REPOSITORY = "mlflow/mlflow"
ISSUE_NUMBER = 42
BOT = "github-actions[bot]"


def report(outcome="ready"):
    return {
        "binding": {"repository": REPOSITORY, "issue_number": ISSUE_NUMBER},
        "outcome": outcome,
        "markdown": f"{MARKER}\n## Triage report",
    }


def issue(**overrides):
    result = {"number": ISSUE_NUMBER, "state": "open", "labels": [{"name": "bug"}]}
    result.update(overrides)
    return result


def test_plan_creates_comment_and_preserves_unrelated_labels_and_assignees():
    current_issue = issue(assignees=[{"login": "someone"}])
    plan = build_publish_plan(
        report(), current_issue, [], repository=REPOSITORY, issue_number=ISSUE_NUMBER, bot_login=BOT
    )

    assert plan.comment_id is None
    assert plan.add_label == "ready"
    assert plan.remove_label is None
    assert current_issue["labels"] == [{"name": "bug"}]
    assert current_issue["assignees"] == [{"login": "someone"}]


def test_plan_updates_only_a_bot_authored_marker_comment():
    comments = [
        {"id": 1, "body": MARKER, "user": {"login": "reporter"}},
        {"id": 2, "body": "old " + MARKER, "user": {"login": BOT}},
        {"id": 3, "body": "new " + MARKER, "user": {"login": BOT}},
    ]
    plan = build_publish_plan(
        report(), issue(), comments, repository=REPOSITORY, issue_number=ISSUE_NUMBER, bot_login=BOT
    )

    assert plan.comment_id == 3


def test_plan_transitions_only_between_the_two_owned_labels():
    plan = build_publish_plan(
        report("needs_committer_feedback"),
        issue(labels=[{"name": "bug"}, {"name": "ready"}, {"name": "needs design"}]),
        [],
        repository=REPOSITORY,
        issue_number=ISSUE_NUMBER,
        bot_login=BOT,
    )

    assert plan.add_label == LABELS["needs_committer_feedback"]
    assert plan.remove_label == "ready"


def test_report_binding_and_untrusted_marker_are_rejected():
    bad_binding = report()
    bad_binding["binding"]["issue_number"] = 43
    with pytest.raises(ValueError, match="binding"):
        validate_report(bad_binding, REPOSITORY, ISSUE_NUMBER)

    bad_marker = report()
    bad_marker["markdown"] = "reporter supplied " + MARKER
    with pytest.raises(ValueError, match="marker"):
        validate_report(bad_marker, REPOSITORY, ISSUE_NUMBER)


def test_closed_issues_and_pull_requests_are_rejected():
    for invalid in (issue(state="closed"), issue(pull_request={})):
        try:
            validate_issue(invalid, ISSUE_NUMBER)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid issue to be rejected")


def test_client_apply_has_only_comment_and_workflow_label_calls():
    calls = []

    class Completed:
        stdout = ""

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    plan = build_publish_plan(
        report("needs_committer_feedback"),
        issue(labels=[{"name": "ready"}, {"name": "bug"}]),
        [],
        repository=REPOSITORY,
        issue_number=ISSUE_NUMBER,
        bot_login=BOT,
    )
    GitHubClient(run).apply(REPOSITORY, ISSUE_NUMBER, plan)

    commands = [call[0] for call in calls]
    assert commands[0][4] == "repos/mlflow/mlflow/issues/42/comments"
    assert commands[1][4].endswith("/labels/ready")
    assert commands[2][4].endswith("/labels")
    assert all("assignees" not in command[4] for command in commands)
