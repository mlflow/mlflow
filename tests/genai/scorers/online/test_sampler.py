import json

import pytest

from mlflow.genai.scorers.builtin_scorers import Completeness, ConversationCompleteness
from mlflow.genai.scorers.online.entities import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler


def make_online_scorer(
    scorer,
    sample_rate: float = 1.0,
    filter_string: str | None = None,
) -> OnlineScorer:
    return OnlineScorer(
        name=scorer.name,
        experiment_id="exp1",
        serialized_scorer=json.dumps(scorer.model_dump()),
        sample_rate=sample_rate,
        filter_string=filter_string,
    )


def test_get_filter_strings_empty():
    sampler = OnlineScorerSampler([])

    assert sampler.get_filter_strings() == set()


def test_get_filter_strings_no_filters():
    configs = [
        make_online_scorer(Completeness()),
        make_online_scorer(ConversationCompleteness()),
    ]
    sampler = OnlineScorerSampler(configs)

    assert sampler.get_filter_strings() == {None}


def test_get_filter_strings_with_filters():
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(), filter_string="tags.model = 'gpt-4'"),
        make_online_scorer(Completeness()),
    ]
    sampler = OnlineScorerSampler(configs)

    result = sampler.get_filter_strings()

    assert result == {None, "tags.env = 'prod'", "tags.model = 'gpt-4'"}


def test_get_filter_strings_deduplicates():
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(ConversationCompleteness(), filter_string="tags.env = 'prod'"),
    ]
    sampler = OnlineScorerSampler(configs)

    assert sampler.get_filter_strings() == {"tags.env = 'prod'"}


def test_get_scorers_for_filter_matching():
    s1 = Completeness()
    s2 = ConversationCompleteness()
    s3 = Completeness()
    configs = [
        make_online_scorer(s1, filter_string="tags.env = 'prod'"),
        make_online_scorer(s2, filter_string="tags.env = 'prod'"),
        make_online_scorer(s3, filter_string="other"),
    ]
    sampler = OnlineScorerSampler(configs)

    result = sampler.get_scorers_for_filter("tags.env = 'prod'", session_level=False)

    assert len(result) == 1
    assert result[0].name == "completeness"


def test_get_scorers_for_filter_no_match():
    configs = [make_online_scorer(Completeness(), filter_string="tags.env = 'prod'")]
    sampler = OnlineScorerSampler(configs)

    result = sampler.get_scorers_for_filter("nonexistent", session_level=False)

    assert result == []


def test_get_scorers_for_filter_session_level():
    configs = [
        make_online_scorer(Completeness()),
        make_online_scorer(ConversationCompleteness()),
    ]
    sampler = OnlineScorerSampler(configs)

    trace_scorers = sampler.get_scorers_for_filter(None, session_level=False)
    session_scorers = sampler.get_scorers_for_filter(None, session_level=True)

    assert [s.name for s in trace_scorers] == ["completeness"]
    assert [s.name for s in session_scorers] == ["conversation_completeness"]


def test_sample_all_selected_at_100_percent():
    configs = [
        make_online_scorer(Completeness(), sample_rate=1.0),
        make_online_scorer(ConversationCompleteness(), sample_rate=1.0),
    ]
    sampler = OnlineScorerSampler(configs)
    scorers = list(sampler._scorers.values())

    result = sampler.sample("entity_123", scorers)

    assert len(result) == 2


def test_sample_none_selected_at_0_percent():
    configs = [make_online_scorer(Completeness(), sample_rate=0.0)]
    sampler = OnlineScorerSampler(configs)
    scorers = list(sampler._scorers.values())

    result = sampler.sample("entity_123", scorers)

    assert result == []


def test_sample_deterministic_by_entity_id():
    configs = [make_online_scorer(Completeness(), sample_rate=0.5)]
    sampler = OnlineScorerSampler(configs)
    scorers = list(sampler._scorers.values())

    results = [sampler.sample("same_entity", scorers) for _ in range(10)]

    assert all(r == results[0] for r in results)


@pytest.mark.parametrize("entity_id", [f"entity_{i}" for i in range(20)])
def test_sample_dense_waterfall_behavior(entity_id):
    high = Completeness(name="high")
    medium = Completeness(name="medium")
    low = Completeness(name="low")
    configs = [
        make_online_scorer(high, sample_rate=0.8),
        make_online_scorer(medium, sample_rate=0.5),
        make_online_scorer(low, sample_rate=0.2),
    ]
    sampler = OnlineScorerSampler(configs)
    scorers = [sampler._scorers["high"], sampler._scorers["medium"], sampler._scorers["low"]]

    result = sampler.sample(entity_id, scorers)

    result_names = [s.name for s in result]
    if "high" not in result_names:
        assert "medium" not in result_names
        assert "low" not in result_names
    if "medium" not in result_names and "high" in result_names:
        assert "low" not in result_names
