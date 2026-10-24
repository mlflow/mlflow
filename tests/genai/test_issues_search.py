import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai._issues_search import _parse_issue_filter


@pytest.mark.parametrize("filter_string", [None, "", "  "])
def test_empty_issue_filter(filter_string):
    assert _parse_issue_filter(filter_string) == {}


def test_issue_filter_translates_dimensions_and_tags():
    assert _parse_issue_filter(
        "status IN ('pending', 'RESOLVED') AND severity = 'high' "
        "AND category IN ('latency', 'custom category') AND title ILIKE '%Bob''s AND retries%' "
        "AND tags.`team.name` IN ('payments', 'api') AND tags.owner IS NOT NULL"
    ) == {
        "statuses": ["PENDING", "RESOLVED"],
        "severities": ["HIGH"],
        "categories": ["latency", "custom category"],
        "query": "Bob's AND retries",
        "tag_filters": [
            {"key": "team.name", "values": ["payments", "api"]},
            {"key": "owner"},
        ],
    }


def test_issue_filter_preserves_custom_categories_and_quoted_tag_values():
    assert _parse_issue_filter(
        "categories = \"Custom.Category\" AND tags.`team owner` = 'Bob''s team'"
    ) == {
        "categories": ["Custom.Category"],
        "tag_filters": [{"key": "team owner", "values": ["Bob's team"]}],
    }


def test_issue_filter_preserves_tag_keys_matching_attribute_aliases():
    assert _parse_issue_filter("tags.categories = 'a' AND tags.category = 'b'") == {
        "tag_filters": [
            {"key": "categories", "values": ["a"]},
            {"key": "category", "values": ["b"]},
        ]
    }


@pytest.mark.parametrize(
    "filter_string",
    [
        "status = 'PENDING' OR status = 'RESOLVED'",
        "AND status = 'PENDING'",
        "status = 'PENDING' AND",
        "status = 'PENDING' AND AND severity = 'HIGH'",
        "status = 'PENDING'; severity = 'HIGH'",
        "status = 'PENDING' severity = 'HIGH'",
        "status = 'PENDING' AND status = 'RESOLVED'",
        "category = 'latency' AND categories = 'correctness'",
        "tags.team = 'api' AND tags.team = 'payments'",
        "status = PENDING",
        "severity = 'unknown'",
        "status = 'deleted'",
        "severity = 'NOT_AN_ISSUE'",
        "status = ('PENDING', 'RESOLVED')",
        "status IN 'PENDING'",
        "status IN ()",
        "status IN ('PENDING',)",
        "status IN ('PENDING', 1)",
        "title = 'foo'",
        "title LIKE '%foo%'",
        "title ILIKE 'foo%'",
        "title ILIKE '%foo'",
        "title ILIKE '%foo_bar%'",
        "title ILIKE '%foo%bar%'",
        "title ILIKE '%'",
        "title IN ('foo')",
        "tags.team != 'api'",
        "tags.team IS NULL",
        "tags.team = 'a' 'b'",
        "source_run_id = 'run'",
        "metrics.latency = '1'",
        5,
    ],
)
def test_issue_filter_rejects_unrepresentable_or_invalid_filters(filter_string):
    with pytest.raises(MlflowException, match="."):
        _parse_issue_filter(filter_string)
