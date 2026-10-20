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
    @mlflow.trace(name=name, span_type=SpanType.AGENT)
    def _run():
        return "done"

    _run()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    if cost_usd is not None:
        trace.info.trace_metadata["mlflow.trace.cost"] = json.dumps({"total_cost": cost_usd})
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


def test_small_labels_are_pooled_rather_than_compared_against_themselves():
    # Four agents of three traces each: no label can be judged alone, but pooled they give a
    # conservative comparison that still catches a runaway.
    traces = []
    for index in range(4):
        traces += _sample(f"agent_{index}", [0.010, 0.011, 0.012])
    traces += _sample("agent_0", [4.00])

    anomalies = analyze_cost_anomalies(traces)

    assert [anomaly.label for anomaly in anomalies] == ["(mixed agents)"]


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
