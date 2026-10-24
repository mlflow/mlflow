from mlflow.entities import IssueResource


def test_issue_resource_rest_roundtrip():
    payload = {
        "name": "experiments/1/issues/iss-payments",
        "issue_id": "iss-payments",
        "title": "Payment retries fail",
        "description": "Retries return an error.",
        "status": "PENDING",
        "severity": "HIGH",
        "categories": ["correctness"],
        "tags": {"team": "payments"},
        "last_seen_time": "2026-10-22T12:34:56.123Z",
        "source_type": "ADHOC_RUN",
        "source_id": "a" * 32,
        "created_by": "reviewer",
        "create_time": "2026-10-22T12:35:00Z",
        "last_updated_by": "owner",
        "update_time": "2026-10-22T12:36:00Z",
        "metadata": {"count": "3"},
    }

    issue = IssueResource.from_dictionary({**payload, "new_server_field": "ignored"})

    assert issue.name == "experiments/1/issues/iss-payments"
    assert issue.title == "Payment retries fail"
    assert issue.to_dictionary() == payload
    serialized = issue.to_dictionary()
    serialized["tags"]["team"] = "changed"
    assert issue.tags == {"team": "payments"}


def test_issue_resource_basic_view_preserves_empty_collections():
    payload = {
        "name": "experiments/1/issues/iss-manual",
        "issue_id": "iss-manual",
        "title": "Manual issue",
        "status": "PENDING",
        "severity": "LOW",
        "source_type": "MANUAL",
        "categories": [],
        "tags": {},
        "metadata": {},
    }

    issue = IssueResource.from_dictionary(payload)

    assert issue.description is None
    assert issue.last_seen_time is None
    assert issue.to_dictionary() == payload
