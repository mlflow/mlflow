import mlflow
from mlflow.entities.span import SpanType
from mlflow.genai.discovery.evidence import (
    _GeneratedEvidence,
    _GeneratedStep,
    build_evidence,
    format_span_inventory,
)


def _trace():
    @mlflow.trace(name="agent", span_type=SpanType.AGENT)
    def _run():
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": "refund window"})
            span.set_outputs({"documents": []})
        with mlflow.start_span(name="generate", span_type=SpanType.LLM) as span:
            span.set_inputs({"prompt": "What is the refund window?"})
            span.set_outputs({"text": "You have 45 days."})
        return "You have 45 days."

    _run()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def _span_id(trace, name):
    return next(span.span_id for span in trace.data.spans if span.name == name)


def _generated(trace, **overrides):
    retrieve = _span_id(trace, "retrieve")
    generate = _span_id(trace, "generate")
    defaults = {
        "summary": "Retrieval returned nothing and the agent answered anyway.",
        "steps": [
            _GeneratedStep(
                span_id=retrieve,
                text="Retrieval returned 0 documents for the refund question.",
                highlights=["0 documents"],
                highlight_tone="error",
            ),
            _GeneratedStep(
                span_id=generate,
                text="The model answered with an invented 45 day window.",
                highlights=["45 day"],
                highlight_tone="error",
            ),
        ],
        "primary_span_id": generate,
    }
    defaults.update(overrides)
    return _GeneratedEvidence(**defaults)


def test_the_span_inventory_lists_ids_so_a_step_can_cite_one():
    trace = _trace()

    inventory = format_span_inventory(trace)

    for span in trace.data.spans:
        assert f"span_id={span.span_id}" in inventory
    assert "name='retrieve'" in inventory
    assert "refund window" in inventory


def test_the_inventory_is_bounded_and_says_what_it_omitted():
    trace = _trace()

    inventory = format_span_inventory(trace, max_spans=1)

    assert "further spans omitted" in inventory


def test_a_sound_walkthrough_becomes_validated_evidence():
    trace = _trace()

    evidence = build_evidence(_generated(trace), trace)

    assert evidence is not None
    assert [step.span_id for step in evidence.steps] == [
        _span_id(trace, "retrieve"),
        _span_id(trace, "generate"),
    ]
    assert evidence.primary_span_id == _span_id(trace, "generate")
    assert evidence.steps[0].highlights == ["0 documents"]


def test_a_step_citing_a_span_the_trace_never_recorded_is_dropped():
    trace = _trace()
    generated = _generated(
        trace,
        steps=[
            _GeneratedStep(
                span_id="span-does-not-exist",
                text="Retrieval returned 0 documents.",
                highlights=["0 documents"],
                highlight_tone="error",
            ),
            _GeneratedStep(
                span_id=_span_id(trace, "generate"),
                text="The model answered with an invented 45 day window.",
                highlights=["45 day"],
                highlight_tone="error",
            ),
        ],
        primary_span_id=None,
    )

    evidence = build_evidence(generated, trace)

    assert [step.span_id for step in evidence.steps] == [_span_id(trace, "generate")]


def test_a_paraphrased_highlight_is_dropped_but_an_exact_one_survives():
    trace = _trace()
    generated = _generated(
        trace,
        steps=[
            _GeneratedStep(
                span_id=_span_id(trace, "retrieve"),
                text="Retrieval returned 0 documents for the refund question.",
                # The first is a paraphrase and emphasizes nothing; the second is exact.
                highlights=["returned zero documents", "0 documents"],
                highlight_tone="error",
            )
        ],
        primary_span_id=None,
    )

    evidence = build_evidence(generated, trace)

    assert evidence.steps[0].highlights == ["0 documents"]


def test_a_step_whose_every_highlight_is_a_paraphrase_is_dropped():
    trace = _trace()
    generated = _generated(
        trace,
        steps=[
            _GeneratedStep(
                span_id=_span_id(trace, "retrieve"),
                text="Retrieval returned 0 documents.",
                highlights=["returned nothing at all"],
                highlight_tone="error",
            )
        ],
        primary_span_id=None,
    )

    assert build_evidence(generated, trace) is None


def test_an_unknown_tone_falls_back_to_warning_rather_than_losing_the_step():
    trace = _trace()
    generated = _generated(
        trace,
        steps=[
            _GeneratedStep(
                span_id=_span_id(trace, "retrieve"),
                text="Retrieval returned 0 documents.",
                highlights=["0 documents"],
                highlight_tone="catastrophic",
            )
        ],
        primary_span_id=None,
    )

    evidence = build_evidence(generated, trace)

    assert evidence.steps[0].highlight_tone == "warning"


def test_a_primary_span_no_step_cites_is_dropped_rather_than_rejected():
    trace = _trace()
    generated = _generated(
        trace,
        steps=[
            _GeneratedStep(
                span_id=_span_id(trace, "retrieve"),
                text="Retrieval returned 0 documents.",
                highlights=["0 documents"],
                highlight_tone="error",
            )
        ],
        primary_span_id=_span_id(trace, "generate"),
    )

    evidence = build_evidence(generated, trace)

    assert evidence is not None
    assert evidence.primary_span_id is None


def test_evidence_with_no_usable_step_degrades_to_nothing_so_the_caller_keeps_the_summary():
    trace = _trace()

    assert build_evidence(_generated(trace, steps=[]), trace) is None


def test_a_missing_summary_falls_back_to_the_callers_prose():
    trace = _trace()

    evidence = build_evidence(
        _generated(trace, summary="  "), trace, fallback_summary="Answered from an empty context."
    )

    assert evidence is not None
    assert evidence.summary == "Answered from an empty context."
