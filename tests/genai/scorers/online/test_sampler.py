import json
import uuid

import pytest

from mlflow.genai.scorers.builtin_scorers import Completeness, ConversationCompleteness
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler


def make_online_scorer(
    scorer,
    sample_rate: float = 1.0,
    filter_string: str | None = None,
) -> OnlineScorer:
    return OnlineScorer(
        name=scorer.name,
        serialized_scorer=json.dumps(scorer.model_dump()),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=sample_rate,
            experiment_id="exp1",
            filter_string=filter_string,
        ),
    )


def test_group_scorers_by_filter_empty():
    sampler = OnlineScorerSampler([])

    assert sampler.group_scorers_by_filter(session_level=False) == {}


def test_group_scorers_by_filter_no_filters():
    configs = [
        make_online_scorer(Completeness()),
        make_online_scorer(ConversationCompleteness()),
    ]
    sampler = OnlineScorerSampler(configs)

    trace_groups = sampler.group_scorers_by_filter(session_level=False)
    session_groups = sampler.group_scorers_by_filter(session_level=True)

    assert set(trace_groups.keys()) == {None}
    assert [s.name for s in trace_groups[None]] == ["completeness"]
    assert set(session_groups.keys()) == {None}
    assert [s.name for s in session_groups[None]] == ["conversation_completeness"]


def test_group_scorers_by_filter_with_filters():
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.model = 'gpt-4'"),
        make_online_scorer(Completeness(name="c3")),
    ]
    sampler = OnlineScorerSampler(configs)

    result = sampler.group_scorers_by_filter(session_level=False)

    assert set(result.keys()) == {None, "tags.env = 'prod'", "tags.model = 'gpt-4'"}
    assert [s.name for s in result["tags.env = 'prod'"]] == ["completeness"]
    assert [s.name for s in result["tags.model = 'gpt-4'"]] == ["c2"]
    assert [s.name for s in result[None]] == ["c3"]


def test_group_scorers_by_filter_multiple_scorers_same_filter():
    configs = [
        make_online_scorer(Completeness(), filter_string="tags.env = 'prod'"),
        make_online_scorer(Completeness(name="c2"), filter_string="tags.env = 'prod'"),
    ]
    sampler = OnlineScorerSampler(configs)

    result = sampler.group_scorers_by_filter(session_level=False)

    assert set(result.keys()) == {"tags.env = 'prod'"}
    assert len(result["tags.env = 'prod'"]) == 2


def test_group_scorers_by_filter_session_level():
    configs = [
        make_online_scorer(Completeness()),
        make_online_scorer(ConversationCompleteness()),
    ]
    sampler = OnlineScorerSampler(configs)

    trace_groups = sampler.group_scorers_by_filter(session_level=False)
    session_groups = sampler.group_scorers_by_filter(session_level=True)

    assert [s.name for s in trace_groups[None]] == ["completeness"]
    assert [s.name for s in session_groups[None]] == ["conversation_completeness"]


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


@pytest.mark.parametrize("entity_id", ["trace-1", "trace-2", "trace-3", "trace-4", "trace-5"])
def test_sample_is_independent_of_scorer_load_order_for_equal_rates(entity_id):
    """
    Scorers with the same sample rate are tie-broken by name. Otherwise the dense-sampling
    waterfall would depend on the order the store returned the configs, and the same
    entity + configuration could be sampled differently across runs or after a restart.
    """
    scorer_a, scorer_b, scorer_c = Completeness(), Completeness(), Completeness()
    scorer_a.name, scorer_b.name, scorer_c.name = "a", "b", "c"
    configs = [
        make_online_scorer(scorer_a, sample_rate=0.5),
        make_online_scorer(scorer_b, sample_rate=0.5),
        make_online_scorer(scorer_c, sample_rate=0.5),
    ]
    forward = OnlineScorerSampler(configs)
    reverse = OnlineScorerSampler(list(reversed(configs)))

    scorers = [forward._scorers[n] for n in ("a", "b", "c")]
    selected_forward = {s.name for s in forward.sample(entity_id, scorers)}
    selected_reverse = {s.name for s in reverse.sample(entity_id, list(reversed(scorers)))}

    assert selected_forward == selected_reverse
