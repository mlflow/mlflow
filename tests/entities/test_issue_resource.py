import pytest

from mlflow.entities import IssueCategory, IssueResource, IssueSeverity, IssueStatus


def test_issue_resource_rest_roundtrip():
    payload = {
        "name": "experiments/1/issues/iss-payments",
        "issue_id": "iss-payments",
        "title": "Payment retries fail",
        "description": "Retries return an error.",
        "status": "PENDING",
        "severity": "HIGH",
        "categories": ["correctness", "payment_processing"],
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
    assert issue.status is IssueStatus.PENDING
    assert issue.severity is IssueSeverity.HIGH
    assert issue.categories == [IssueCategory.CORRECTNESS, "payment_processing"]
    assert issue.categories[0] is IssueCategory.CORRECTNESS
    assert issue.to_dictionary() == payload
    serialized = issue.to_dictionary()
    serialized["tags"]["team"] = "changed"
    assert issue.tags == {"team": "payments"}


@pytest.mark.parametrize("category", list(IssueCategory))
def test_issue_resource_builtin_category_roundtrip(category):
    issue = IssueResource(
        name="experiments/1/issues/iss-manual",
        issue_id="iss-manual",
        title="Manual issue",
        status=IssueStatus.PENDING,
        severity=IssueSeverity.MEDIUM,
        categories=[category, "custom_category"],
    )

    serialized = issue.to_dictionary()

    assert serialized["status"] == "PENDING"
    assert serialized["severity"] == "MEDIUM"
    assert serialized["categories"] == [category.value, "custom_category"]
    restored = IssueResource.from_dictionary(serialized)
    assert restored == issue
    assert restored.categories[0] is category


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
