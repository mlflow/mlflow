import pytest

from mlflow.exceptions import MlflowException
from mlflow.insights.models import EvidenceEntry
from mlflow.insights.models.base import extract_unique_trace_ids
from mlflow.insights.utils import normalize_evidence


@pytest.mark.parametrize(
    ("evidence", "for_issue", "expected_supports"),
    [
        # Hypothesis evidence with explicit supports values
        ([{"trace_id": "t1", "rationale": "r1", "supports": True}], False, [True]),
        ([{"trace_id": "t1", "rationale": "r1", "supports": False}], False, [False]),
        # Hypothesis evidence defaults to True when supports not specified
        ([{"trace_id": "t1", "rationale": "r1"}], False, [True]),
        # Issue evidence always has supports=None
        ([{"trace_id": "t1", "rationale": "r1", "supports": True}], True, [None]),
        ([{"trace_id": "t1", "rationale": "r1"}], True, [None]),
        # Multiple entries
        (
            [
                {"trace_id": "t1", "rationale": "r1", "supports": True},
                {"trace_id": "t2", "rationale": "r2", "supports": False},
                {"trace_id": "t3", "rationale": "r3"},
            ],
            False,
            [True, False, True],
        ),
    ],
)
def test_normalize_evidence_supports_field(evidence, for_issue, expected_supports):
    result = normalize_evidence(evidence, for_issue=for_issue)
    assert [e.supports for e in result] == expected_supports


def test_normalize_evidence_mixed_input_types():
    evidence = [
        {"trace_id": "t1", "rationale": "dict"},
        EvidenceEntry(trace_id="t2", rationale="object", supports=False),
    ]
    result = normalize_evidence(evidence)

    assert len(result) == 2
    assert all(isinstance(e, EvidenceEntry) for e in result)
    assert result[0].trace_id == "t1"
    assert result[1].trace_id == "t2"


def test_normalize_evidence_strips_whitespace():
    evidence = [{"trace_id": "  t1  ", "rationale": "  test  "}]
    result = normalize_evidence(evidence)

    assert result[0].trace_id == "t1"
    assert result[0].rationale == "test"


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (None, []),
        ([], []),
    ],
)
def test_normalize_evidence_empty(evidence, expected):
    assert normalize_evidence(evidence) == expected


@pytest.mark.parametrize(
    ("invalid_evidence", "error_pattern"),
    [
        ([{}], "trace_id"),
        ([{"trace_id": "t1"}], "rationale"),
        ([{"trace_id": "", "rationale": "test"}], "cannot be empty"),
        ([{"trace_id": "test", "rationale": ""}], "cannot be empty"),
        ([123], "must be a dict or EvidenceEntry"),
    ],
)
def test_normalize_evidence_invalid(invalid_evidence, error_pattern):
    with pytest.raises(MlflowException, match=error_pattern):
        normalize_evidence(invalid_evidence)


@pytest.mark.parametrize(
    ("evidence_data", "expected_ids"),
    [
        # Basic extraction
        ([("t1", "r1"), ("t2", "r2"), ("t3", "r3")], ["t1", "t2", "t3"]),
        # Deduplication preserves first occurrence order
        ([("t1", "r1"), ("t2", "r2"), ("t1", "r3")], ["t1", "t2"]),
        # Order preservation
        ([("c", "r1"), ("a", "r2"), ("b", "r3")], ["c", "a", "b"]),
        # Empty
        ([], []),
    ],
)
def test_extract_unique_trace_ids(evidence_data, expected_ids):
    evidence = [EvidenceEntry(trace_id=tid, rationale=rat) for tid, rat in evidence_data]
    assert extract_unique_trace_ids(evidence) == expected_ids
