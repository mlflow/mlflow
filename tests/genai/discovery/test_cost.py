import json
import random
import statistics
from unittest import mock

import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.discovery.utils import (
    analyze_cost_anomalies,
    get_agent_label,
)


def _trace(*, name, cost_usd=None):
    """
    A trace carrying an exact cost.

    The cost is injected into the fetched trace's metadata rather than derived from token
    usage on an LLM span. Both go through ``TraceInfo.cost``, but the derived path prices
    tokens with LiteLLM's catalog, so in an environment without ``litellm`` every trace would
    come back unpriced and these tests would pass vacuously on empty input. Injection keeps
    them independent of that, and the assertion below keeps the injection honest: if the
    attribute these tests rely on ever stops being read from this dict, they fail loudly here
    instead of silently asserting nothing.
    """

    @mlflow.trace(name=name, span_type=SpanType.AGENT)
    def _run():
        return "done"

    _run()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    if cost_usd is not None:
        trace.info.trace_metadata["mlflow.trace.cost"] = json.dumps({"total_cost": cost_usd})
        assert (trace.info.cost or {}).get("total_cost") == cost_usd
    return trace


def _sample(name, costs):
    return [_trace(name=name, cost_usd=cost) for cost in costs]


def _ordinary_costs(count, seed=3, spread=0.5):
    # Agent spend is right-skewed, so the healthy case is lognormal rather than uniform.
    rng = random.Random(seed)
    return [rng.lognormvariate(0, spread) * 0.02 for _ in range(count)]


def test_the_agent_label_comes_from_the_trace_name():
    assert get_agent_label(_trace(name="summarizer", cost_usd=0.01)) == "summarizer"


def test_an_ordinary_spread_of_spend_files_nothing():
    # The question this module exists to answer correctly: a sample always has a most
    # expensive trace, and that alone is not an anomaly.
    assert analyze_cost_anomalies(_sample("agent", _ordinary_costs(20))) == []


def test_a_wide_but_ordinary_spread_still_files_nothing():
    # A workload whose spend naturally varies raises its own bar rather than reporting its
    # own tail as a finding.
    assert analyze_cost_anomalies(_sample("agent", _ordinary_costs(20, spread=1.2))) == []


def test_a_runaway_trace_is_flagged_against_its_peers():
    costs = _ordinary_costs(19)
    runaway = statistics.median(costs) * 40
    traces = _sample("agent", [*costs, runaway])

    anomalies = analyze_cost_anomalies(traces)

    assert len(anomalies) == 1
    assert anomalies[0].cost_usd == runaway
    assert anomalies[0].label == "agent"
    assert anomalies[0].peer_count == 20
    # The finding says what the spend is compared against, or it is not actionable.
    description = anomalies[0].describe()
    assert "the typical" in description
    assert "20 priced traces" in description


def test_a_trace_expensive_only_for_a_tight_workload_is_still_an_anomaly():
    # 8x the typical trace is not a large multiple in absolute terms, but it is far outside
    # a workload whose spend barely varies -- the bar adapts to the sample.
    costs = _ordinary_costs(19, spread=0.1)
    traces = _sample("agent", [*costs, statistics.median(costs) * 8])

    assert len(analyze_cost_anomalies(traces)) == 1


def test_spend_is_compared_within_a_label_not_across_the_experiment():
    # A summarizer that simply costs more than a retriever is not an anomaly.
    traces = _sample("retriever", _ordinary_costs(12, seed=1))
    traces += _sample("summarizer", [cost * 100 for cost in _ordinary_costs(12, seed=2)])

    assert analyze_cost_anomalies(traces) == []


def test_a_runaway_is_found_in_the_label_it_belongs_to():
    cheap = _ordinary_costs(12, seed=1)
    traces = _sample("retriever", cheap)
    traces += _sample("summarizer", [cost * 100 for cost in _ordinary_costs(12, seed=2)])
    traces += _sample("retriever", [statistics.median(cheap) * 50])

    anomalies = analyze_cost_anomalies(traces)

    assert [anomaly.label for anomaly in anomalies] == ["retriever"]


def test_a_sample_too_small_to_judge_files_nothing():
    traces = _sample("agent", [0.01, 0.01, 0.01, 0.80])

    assert analyze_cost_anomalies(traces) == []


def test_a_group_where_most_traces_qualify_is_a_distribution_not_an_outlier():
    # Half the sample an order of magnitude above the other half is two populations, not ten
    # anomalies, so nothing is reported rather than everything.
    traces = _sample("agent", [0.01] * 10 + [5.00] * 10)

    assert analyze_cost_anomalies(traces) == []


def test_identical_spend_has_no_outlier_to_find():
    assert analyze_cost_anomalies(_sample("agent", [0.02] * 12)) == []


def test_missing_cost_is_excluded_rather_than_read_as_zero_spend():
    # Were unpriced traces counted as $0, the one real cost would look anomalous.
    traces = [_trace(name="agent") for _ in range(12)]
    traces.append(_trace(name="agent", cost_usd=0.50))

    assert analyze_cost_anomalies(traces) == []


def test_a_label_without_enough_priced_peers_is_not_judged():
    # Four agents of three traces each, one carrying a runaway. Nothing is reported: no label
    # has the peers to judge it. This test previously asserted the opposite -- that the labels
    # were pooled and the runaway found -- which worked only because every label here costs
    # about the same. See the test below for the case that breaks.
    traces = []
    for index in range(4):
        traces += _sample(f"agent_{index}", [0.010, 0.011, 0.012])
    traces += _sample("agent_0", [4.00])

    assert analyze_cost_anomalies(traces) == []


def test_an_ordinary_trace_of_a_costlier_agent_is_not_an_anomaly():
    # Reported in review. Pooling undersampled labels compares a trace against whatever other
    # agents happen to be in the sample, so these two perfectly ordinary reasoner traces were
    # reported at "75x the typical" purely because the other four agents are cheap.
    traces = []
    for label in ("retriever", "classifier", "formatter", "router"):
        traces += _sample(label, [0.010 + i * 0.001 for i in range(8)])
    reasoner = _sample("reasoner", [1.00, 1.05])

    assert analyze_cost_anomalies(traces + reasoner) == []

    # The decisive part: giving those same two traces ordinary same-agent peers used to remove
    # the findings. A verdict that flips on the rest of the sample was not about these traces.
    peers = _sample("reasoner", [0.95 + i * 0.01 for i in range(8)])
    assert analyze_cost_anomalies(traces + reasoner + peers) == []


def test_a_runaway_is_still_caught_once_its_own_label_has_peers():
    # The flip side of skipping undersampled labels: detection must survive for a label that
    # does have peers, or the fix above would have turned the feature off.
    traces = _sample("agent", [0.010 + i * 0.0005 for i in range(19)])
    traces += _sample("agent", [0.40])

    anomalies = analyze_cost_anomalies(traces)

    assert [anomaly.label for anomaly in anomalies] == ["agent"]
    assert anomalies[0].cost_usd == 0.40


def test_the_anomalous_fraction_cutoff_is_exclusive_at_ten_percent():
    # Reported in review: the 10% cutoff boundary was untested. Two qualifying traces among 19
    # exceed it (2 > 1.9) and are suppressed; one more ordinary trace brings it to 2 > 2.0,
    # which is false, so both appear. Pinned so the boundary cannot move unnoticed.
    base = [0.010 + i * 0.0005 for i in range(17)]
    runaways = [0.40, 0.42]

    nineteen = _sample("agent", base) + _sample("agent", runaways)
    assert len(nineteen) == 19
    assert analyze_cost_anomalies(nineteen) == []

    twenty = nineteen + _sample("agent", [0.011])
    assert len(twenty) == 20
    assert len(analyze_cost_anomalies(twenty)) == 2


def test_an_empty_sample_produces_no_findings():
    assert analyze_cost_anomalies([]) == []


def test_a_trace_whose_cost_cannot_be_read_is_excluded_not_fatal():
    traces = [mock.MagicMock(spec="Trace") for _ in range(12)]

    assert analyze_cost_anomalies(traces) == []


# ---- cost findings reaching triage ----


def _runaway_sample():
    costs = _ordinary_costs(19)
    traces = _sample("agent", costs)
    expensive = _trace(name="agent", cost_usd=statistics.median(costs) * 40)
    traces.append(expensive)
    return traces, expensive


def test_a_cost_only_anomaly_becomes_a_finding_on_an_otherwise_passing_trace():
    from mlflow.genai.discovery.constants import CATEGORY_COST
    from mlflow.genai.discovery.entities import _TriageResult
    from mlflow.genai.discovery.pipeline import _merge_cost_findings

    traces, expensive = _runaway_sample()
    # Triage found nothing: the trace answered the user and returned OK.
    triage = _TriageResult(failing_traces=[], rationale_map={}, categories_map={})

    added = _merge_cost_findings(
        triage, analyze_cost_anomalies(traces), {t.info.trace_id: t for t in traces}
    )

    assert added == 1
    assert [t.info.trace_id for t in triage.failing_traces] == [expensive.info.trace_id]
    rationale = triage.rationale_map[expensive.info.trace_id]
    assert rationale.startswith("[cost anomaly]")
    assert "the typical" in rationale
    assert triage.categories_map[expensive.info.trace_id] == [CATEGORY_COST]


def test_a_quality_failure_keeps_its_rationale_when_cost_is_added():
    from mlflow.genai.discovery.entities import _TriageResult
    from mlflow.genai.discovery.pipeline import _merge_cost_findings

    traces, expensive = _runaway_sample()
    triage = _TriageResult(
        failing_traces=[expensive],
        rationale_map={expensive.info.trace_id: "answered the wrong question"},
        categories_map={expensive.info.trace_id: ["correctness"]},
    )

    added = _merge_cost_findings(
        triage, analyze_cost_anomalies(traces), {t.info.trace_id: t for t in traces}
    )

    assert added == 0
    assert len(triage.failing_traces) == 1
    rationale = triage.rationale_map[expensive.info.trace_id]
    assert "answered the wrong question" in rationale
    assert "[cost anomaly]" in rationale
    assert triage.categories_map[expensive.info.trace_id] == ["correctness", "cost"]


def test_a_runaway_among_unanimous_peers_is_not_allowed_to_set_its_own_threshold():
    # Reported in review: when the peers agree on one figure the deviation estimated from the
    # whole group grows with the candidate, so the score saturates and no multiple is ever
    # extreme enough. Nine identical traces and one at 40x must flag.
    traces = _sample("agent", [0.01] * 9 + [0.40])

    anomalies = analyze_cost_anomalies(traces)

    assert [anomaly.cost_usd for anomaly in anomalies] == [0.40]
    # And raising it further must not make it less anomalous.
    assert len(analyze_cost_anomalies(_sample("agent", [0.01] * 9 + [400.0]))) == 1


def test_two_runaways_among_unanimous_peers_are_both_reported():
    traces = _sample("agent", [0.01] * 18 + [0.40, 0.44])

    anomalies = analyze_cost_anomalies(traces)

    assert sorted(anomaly.cost_usd for anomaly in anomalies) == [0.40, 0.44]


def test_a_runaway_is_caught_at_the_minimum_sample_size():
    costs = _ordinary_costs(9)
    traces = _sample("agent", [*costs, statistics.median(costs) * 40])

    assert len(analyze_cost_anomalies(traces)) == 1


def test_two_runaways_among_twenty_traces_are_both_reported():
    costs = _ordinary_costs(18)
    runaway = statistics.median(costs) * 40
    traces = _sample("agent", [*costs, runaway, runaway * 1.1])

    assert len(analyze_cost_anomalies(traces)) == 2


def test_an_explicit_zero_cost_is_not_mistaken_for_a_missing_one():
    # A recorded $0 is priced information; it is excluded because a log-scale comparison has
    # no position for zero, not because the trace failed to record its spend.
    from mlflow.genai.discovery.utils import _trace_cost

    assert _trace_cost(_trace(name="agent", cost_usd=0.0)) == 0.0
    assert _trace_cost(_trace(name="agent")) is None
