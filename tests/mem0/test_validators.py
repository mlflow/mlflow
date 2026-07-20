import json
from pathlib import Path

import pytest

from tests.mem0 import validators

FIXTURE = Path(__file__).parent / "fixtures" / "memory_operations.jsonl"


def _load_cases():
    with FIXTURE.open() as f:
        return [json.loads(line) for line in f if line.strip()]


CASES = _load_cases()
CASES_BY_ID = {c["case_id"]: c for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_classify_matches_expected_case_class(case):
    assert validators.classify(case) == case["expected"]["case_class"]


def test_positive_case_is_valid_and_joined():
    case = CASES_BY_ID["valid_search_used_memory"]
    assert validators.load_receipt(case) == "valid"
    assert validators.usefulness_claim(case) == "joined"
    assert validators.classify(case) == "valid"


def test_wrong_scope_detected():
    assert validators.detect_wrong_scope(CASES_BY_ID["wrong_scope_retrieval"]) is True


def test_stale_memory_detected():
    assert validators.detect_stale_memory(CASES_BY_ID["stale_memory_used"]) is True


def test_unjoinable_detected():
    case = CASES_BY_ID["unjoinable_memory"]
    assert validators.detect_unjoinable(case) is True
    assert validators.usefulness_claim(case) == "load_only"


def test_requires_raw_payload_only_for_the_raw_case():
    for case in CASES:
        expected = case["case_id"] == "raw_payload_required"
        assert validators.requires_raw_payload(case) is expected


def test_no_case_carries_a_raw_memory_body():
    # The privacy boundary, made executable: the fixture stores ids/hashes/counts
    # only. A raw ``"memory"`` field (the extracted fact Mem0 returns) must never
    # appear — if it did, a validator could cheat by reading it.
    for case in CASES:
        for event in case["events"]:
            assert "memory" not in event
            for result in event.get("results", []):
                assert "memory" not in result
