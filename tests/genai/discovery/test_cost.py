import json

import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.discovery.cost import (
    ANOMALY_SIGMA,
    EXCLUDED_FEW_PEERS,
    EXCLUDED_NO_COST,
    analyze_cost_anomalies,
    get_agent_label,
)


def _trace(*, name, cost_usd=None, tool_calls=0, repeated_tool=None, repeats=0):
    @mlflow.trace(name=name, span_type=SpanType.AGENT)
    def _run():
        for index in range(tool_calls):
            with mlflow.start_span(name=f"tool_{index}", span_type=SpanType.TOOL) as span:
                span.set_outputs({"result": "ok"})
        for _ in range(repeats):
            with mlflow.start_span(name=repeated_tool, span_type=SpanType.TOOL) as span:
                span.set_outputs({"result": "retry"})
        return "done"

    _run()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    if cost_usd is not None:
        trace.info.trace_metadata["mlflow.trace.cost"] = json.dumps({"total_cost": cost_usd})
    return trace


def test_the_agent_label_comes_from_the_trace_name():
    assert get_agent_label(_trace(name="summarizer", cost_usd=0.01)) == "summarizer"


def test_spend_is_baselined_within_a_label_not_across_the_experiment():
    # A summarizer that is simply more expensive than a retriever is not an anomaly: each
    # label is compared against its own peers.
    traces = [_trace(name="retriever", cost_usd=0.01) for _ in range(6)]
    traces += [_trace(name="summarizer", cost_usd=2.00) for _ in range(6)]

    analysis = analyze_cost_anomalies(traces)

    assert set(analysis.baselines) == {"retriever", "summarizer"}
    assert analysis.anomalies == []


def test_a_trace_anomalous_for_its_own_label_is_flagged():
    traces = [_trace(name="retriever", cost_usd=0.010 + index * 0.0005) for index in range(8)]
    traces.append(_trace(name="retriever", cost_usd=0.90, tool_calls=12))

    analysis = analyze_cost_anomalies(traces)

    assert [anomaly.cost_usd for anomaly in analysis.anomalies] == [0.90]
    anomaly = analysis.anomalies[0]
    assert anomaly.sigma >= ANOMALY_SIGMA
    # The finding says what the spend is compared against, or it is not actionable.
    assert "against a typical" in anomaly.describe()
    assert anomaly.label == "retriever"


def test_a_bounded_driver_is_quantified_and_graded_medium():
    traces = [
        _trace(name="agent", cost_usd=0.010 + index * 0.0004, tool_calls=2) for index in range(8)
    ]
    traces.append(_trace(name="agent", cost_usd=0.80, tool_calls=9))

    analysis = analyze_cost_anomalies(traces)

    anomaly = analysis.anomalies[0]
    assert anomaly.drivers
    assert any("tool calls against a typical" in driver.description for driver in anomaly.drivers)
    assert anomaly.has_unbounded_driver is False
    assert anomaly.severity == "medium"


def test_a_repeated_identical_call_reads_as_unbounded_and_grades_high():
    traces = [
        _trace(name="agent", cost_usd=0.010 + index * 0.0004, tool_calls=2) for index in range(8)
    ]
    traces.append(
        _trace(name="agent", cost_usd=0.80, tool_calls=2, repeated_tool="search", repeats=6)
    )

    analysis = analyze_cost_anomalies(traces)

    anomaly = analysis.anomalies[0]
    assert any("repeats 'search' 6 times" in driver.description for driver in anomaly.drivers)
    assert anomaly.has_unbounded_driver is True
    assert anomaly.severity == "high"


def test_spend_above_baseline_but_below_the_threshold_is_reported_as_low():
    traces = [_trace(name="agent", cost_usd=0.010 + index * 0.001) for index in range(8)]
    traces.append(_trace(name="agent", cost_usd=0.0205))

    analysis = analyze_cost_anomalies(traces)

    assert analysis.anomalies == []
    assert [entry.severity for entry in analysis.elevated] == ["low"]


def test_a_label_with_too_few_peers_is_excluded_not_compared():
    traces = [_trace(name="rare", cost_usd=0.01), _trace(name="rare", cost_usd=5.00)]

    analysis = analyze_cost_anomalies(traces)

    assert analysis.baselines == {}
    assert analysis.anomalies == []
    assert set(analysis.excluded.values()) == {EXCLUDED_FEW_PEERS}
    assert "fewer than" in analysis.describe_limits()


def test_missing_cost_is_excluded_rather_than_treated_as_zero_spend():
    # Were the unpriced traces counted as $0, the one real cost would look anomalous.
    traces = [_trace(name="agent") for _ in range(6)]
    traces.append(_trace(name="agent", cost_usd=0.50))

    analysis = analyze_cost_anomalies(traces)

    assert analysis.anomalies == []
    assert list(analysis.excluded.values()).count(EXCLUDED_NO_COST) == 6
    assert "no recorded cost" in analysis.describe_limits()


def test_a_single_expensive_trace_does_not_raise_its_own_baseline():
    traces = [
        _trace(name="agent", cost_usd=0.010 + index * 0.0004, tool_calls=2) for index in range(8)
    ]
    traces.append(_trace(name="agent", cost_usd=0.80, tool_calls=9))

    analysis = analyze_cost_anomalies(traces)

    anomaly = analysis.anomalies[0]
    # The driver compares against the peers, so the flagged trace is absent from the typical
    # call count it is measured against.
    assert any("typical 2" in driver.description for driver in anomaly.drivers)


def test_the_same_problem_in_a_different_sample_quotes_a_different_baseline():
    # Guards the dedup rule that figures never decide identity: both runs find the same
    # mechanism, and neither baseline is the issue's identity.
    cheap = [_trace(name="agent", cost_usd=0.010, tool_calls=2) for _ in range(8)]
    cheap.append(_trace(name="agent", cost_usd=0.80, tool_calls=9))
    pricey = [_trace(name="agent", cost_usd=0.100, tool_calls=2) for _ in range(8)]
    pricey.append(_trace(name="agent", cost_usd=8.00, tool_calls=9))

    first = analyze_cost_anomalies(cheap).anomalies[0]
    second = analyze_cost_anomalies(pricey).anomalies[0]

    assert first.baseline.median_usd != second.baseline.median_usd
    assert first.describe() != second.describe()
    drivers = {driver.description for driver in first.drivers}
    assert drivers == {driver.description for driver in second.drivers}


def test_an_empty_sample_produces_no_findings():
    analysis = analyze_cost_anomalies([])

    assert analysis.baselines == {}
    assert analysis.anomalies == []
    assert analysis.describe_limits() == ""


# ---- cost findings reaching triage ----


def _adequacy_for(traces, cost_judgeable=True):
    from mlflow.genai.discovery.coverage import TraceDataAdequacy

    return {
        trace.info.trace_id: TraceDataAdequacy(
            trace_id=trace.info.trace_id, cost_judgeable=cost_judgeable
        )
        for trace in traces
    }


def test_a_cost_only_anomaly_becomes_a_finding_on_an_otherwise_passing_trace():
    from mlflow.genai.discovery.constants import CATEGORY_COST
    from mlflow.genai.discovery.entities import _TriageResult
    from mlflow.genai.discovery.pipeline import _merge_cost_findings

    traces = [_trace(name="agent", cost_usd=0.010 + i * 0.0004, tool_calls=2) for i in range(8)]
    expensive = _trace(name="agent", cost_usd=0.80, tool_calls=9)
    traces.append(expensive)
    analysis = analyze_cost_anomalies(traces)
    # Triage found nothing: the trace answered the user and returned OK.
    triage = _TriageResult(failing_traces=[], rationale_map={}, categories_map={})

    added = _merge_cost_findings(
        triage, analysis, {t.info.trace_id: t for t in traces}, _adequacy_for(traces)
    )

    assert added == 1
    assert [t.info.trace_id for t in triage.failing_traces] == [expensive.info.trace_id]
    rationale = triage.rationale_map[expensive.info.trace_id]
    assert rationale.startswith("[cost anomaly]")
    assert "against a typical" in rationale
    assert triage.categories_map[expensive.info.trace_id] == [CATEGORY_COST]


def test_a_quality_failure_keeps_its_rationale_when_cost_is_added():
    from mlflow.genai.discovery.entities import _TriageResult
    from mlflow.genai.discovery.pipeline import _merge_cost_findings

    traces = [_trace(name="agent", cost_usd=0.010 + i * 0.0004, tool_calls=2) for i in range(8)]
    expensive = _trace(name="agent", cost_usd=0.80, tool_calls=9)
    traces.append(expensive)
    analysis = analyze_cost_anomalies(traces)
    triage = _TriageResult(
        failing_traces=[expensive],
        rationale_map={expensive.info.trace_id: "answered the wrong question"},
        categories_map={expensive.info.trace_id: ["correctness"]},
    )

    added = _merge_cost_findings(
        triage, analysis, {t.info.trace_id: t for t in traces}, _adequacy_for(traces)
    )

    assert added == 0
    assert len(triage.failing_traces) == 1
    rationale = triage.rationale_map[expensive.info.trace_id]
    assert "answered the wrong question" in rationale
    assert "[cost anomaly]" in rationale
    assert triage.categories_map[expensive.info.trace_id] == ["correctness", "cost"]


def test_an_anomaly_on_a_trace_with_incomplete_usage_is_not_turned_into_a_finding():
    from mlflow.genai.discovery.entities import _TriageResult
    from mlflow.genai.discovery.pipeline import _merge_cost_findings

    traces = [_trace(name="agent", cost_usd=0.010 + i * 0.0004, tool_calls=2) for i in range(8)]
    traces.append(_trace(name="agent", cost_usd=0.80, tool_calls=9))
    analysis = analyze_cost_anomalies(traces)
    triage = _TriageResult(failing_traces=[], rationale_map={}, categories_map={})

    added = _merge_cost_findings(
        triage,
        analysis,
        {t.info.trace_id: t for t in traces},
        _adequacy_for(traces, cost_judgeable=False),
    )

    assert added == 0
    assert triage.failing_traces == []
