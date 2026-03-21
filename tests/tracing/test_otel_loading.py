import uuid
from pathlib import Path

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource as OTelSDKResource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util._once import Once

import mlflow
from mlflow.entities import SpanStatusCode
from mlflow.entities.assessment import AssessmentSource, Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.server import handlers
from mlflow.server.fastapi_app import app
from mlflow.server.handlers import initialize_backend_stores
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator
from mlflow.tracing.otel.translation.genai_semconv import GenAiTranslator
from mlflow.tracing.otel.translation.open_inference import OpenInferenceTranslator
from mlflow.tracing.otel.translation.traceloop import TraceloopTranslator
from mlflow.tracing.provider import _get_trace_exporter
from mlflow.tracing.utils import encode_trace_id
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.helper_functions import get_safe_port
from tests.tracking.integration_test_utils import ServerThread

if IS_TRACING_SDK_ONLY:
    pytest.skip("OTel get_trace tests require full MLflow server", allow_module_level=True)


@pytest.fixture
def mlflow_server(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path.joinpath("artifacts").as_uri()

    # Force-reset backend stores before each test
    handlers._tracking_store = None
    handlers._model_registry_store = None
    initialize_backend_stores(db_uri, default_artifact_root=artifact_uri)

    with ServerThread(app, get_safe_port()) as url:
        yield url


@pytest.fixture(autouse=True)
def tracking_uri_setup(mlflow_server):
    with _use_tracking_uri(mlflow_server):
        yield


@pytest.fixture(params=[True, False])
def is_async(request, monkeypatch):
    monkeypatch.setenv("MLFLOW_ASYNC_TRACE_LOGGING", "true" if request.param else "false")


def _flush_async_logging():
    exporter = _get_trace_exporter()
    assert hasattr(exporter, "_async_queue"), "Async queue is not initialized"
    exporter._async_queue.flush(terminate=True)


def create_tracer(mlflow_server: str, experiment_id: str, service_name: str = "test-service"):
    resource = OTelSDKResource.create({"service.name": service_name, "service.version": "1.0.0"})
    tracer_provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=f"{mlflow_server}/v1/traces",
        headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
        timeout=10,
    )

    span_processor = SimpleSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Reset the global tracer provider
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_trace._TRACER_PROVIDER = None
    otel_trace.set_tracer_provider(tracer_provider)

    return otel_trace.get_tracer(__name__)


def test_get_trace_for_otel_sent_span(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-get-trace-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "test-service-get-trace")

    # Create a span with various attributes to test conversion
    with tracer.start_as_current_span("otel-test-span") as span:
        span.set_attribute("test.string", "string-value")
        span.set_attribute("test.number", 42)
        span.set_attribute("test.boolean", True)
        span.set_attribute("operation.type", "llm_request")

        # Capture the OTel trace ID
        otel_trace_id = span.get_span_context().trace_id
        assert span.get_span_context().is_valid
        assert otel_trace_id != 0

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) > 0, "No traces found in the database"

    trace_id = traces[0].info.trace_id
    retrieved_trace = mlflow.get_trace(trace_id)

    assert retrieved_trace.info.trace_id == trace_id
    assert retrieved_trace.info.trace_location.mlflow_experiment.experiment_id == experiment_id

    assert len(retrieved_trace.data.spans) == 1
    span = retrieved_trace.data.spans[0]

    assert span.name == "otel-test-span"
    assert span.trace_id == trace_id
    # OTel spans default to UNSET status if not explicitly set
    assert span.status.status_code == SpanStatusCode.UNSET

    # Verify attributes were converted correctly
    assert span.attributes["test.string"] == "string-value"
    assert span.attributes["test.number"] == 42
    assert span.attributes["test.boolean"] is True
    assert span.attributes["operation.type"] == "llm_request"

    # Verify the trace ID matches the expected format
    expected_trace_id = f"tr-{encode_trace_id(otel_trace_id)}"
    assert trace_id == expected_trace_id


def test_get_trace_for_otel_nested_spans(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-nested-spans-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "nested-test-service")

    # Create nested spans
    with tracer.start_as_current_span("parent-span") as parent_span:
        parent_span.set_attribute("span.level", "parent")

        with tracer.start_as_current_span("child-span") as child_span:
            child_span.set_attribute("span.level", "child")
            child_span.set_attribute("child.operation", "process_data")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) > 0, "No traces found in the database"

    trace_id = traces[0].info.trace_id
    retrieved_trace = mlflow.get_trace(trace_id)

    assert len(retrieved_trace.data.spans) == 2

    spans_by_name = {span.name: span for span in retrieved_trace.data.spans}

    assert "parent-span" in spans_by_name
    assert "child-span" in spans_by_name

    parent_span = spans_by_name["parent-span"]
    child_span = spans_by_name["child-span"]

    assert parent_span.attributes["span.level"] == "parent"
    assert parent_span.parent_id is None  # Root span has no parent

    assert child_span.attributes["span.level"] == "child"
    assert child_span.attributes["child.operation"] == "process_data"
    assert child_span.parent_id == parent_span.span_id  # Child should reference parent


def test_get_trace_with_otel_span_events(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-events-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "events-test-service")

    # Create span with events using OTel SDK
    with tracer.start_as_current_span("span-with-events") as span:
        span.add_event("test_event", attributes={"event.type": "processing"})

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )

    trace_id = traces[0].info.trace_id
    retrieved_trace = mlflow.get_trace(trace_id)

    assert len(retrieved_trace.data.spans) == 1
    retrieved_span = retrieved_trace.data.spans[0]

    assert retrieved_span.name == "span-with-events"
    assert len(retrieved_span.events) == 1
    event = retrieved_span.events[0]
    assert event.name == "test_event"
    assert event.attributes["event.type"] == "processing"


def test_get_trace_nonexistent_otel_trace(mlflow_server: str):
    # Create a fake trace ID in OTel format
    fake_otel_trace_id = uuid.uuid4().hex
    fake_trace_id = f"tr-{fake_otel_trace_id}"

    # MLflow get_trace returns None for non-existent traces
    trace = mlflow.get_trace(fake_trace_id)
    assert trace is None


def test_get_trace_with_otel_span_status(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-status-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "status-test-service")

    # Create span with error status using OTel SDK
    with tracer.start_as_current_span("error-span") as span:
        span.set_status(Status(StatusCode.ERROR, "Something went wrong"))

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )

    trace_id = traces[0].info.trace_id
    retrieved_trace = mlflow.get_trace(trace_id)

    assert len(retrieved_trace.data.spans) == 1
    retrieved_span = retrieved_trace.data.spans[0]

    assert retrieved_span.name == "error-span"
    assert retrieved_span.status.status_code == SpanStatusCode.ERROR
    assert "Something went wrong" in retrieved_span.status.description


def test_set_trace_tag_on_otel_trace(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-tag-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "tag-test-service")

    with tracer.start_as_current_span("tagged-span") as span:
        span.set_attribute("test.attribute", "value")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )
    trace_id = traces[0].info.trace_id

    mlflow.set_trace_tag(trace_id, "environment", "test")
    mlflow.set_trace_tag(trace_id, "version", "1.0.0")

    retrieved_trace = mlflow.get_trace(trace_id)
    assert retrieved_trace.info.tags["environment"] == "test"
    assert retrieved_trace.info.tags["version"] == "1.0.0"


def test_log_expectation_on_otel_trace(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-expectation-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "expectation-test-service")

    # Create a span that represents a question-answer scenario
    with tracer.start_as_current_span("qa-span") as span:
        span.set_attribute("question", "What is MLflow?")
        span.set_attribute("answer", "MLflow is an open-source ML platform")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )
    trace_id = traces[0].info.trace_id

    expectation_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="test_user@example.com"
    )

    logged_assessment = mlflow.log_expectation(
        trace_id=trace_id,
        name="expected_answer",
        value="MLflow is an open-source machine learning platform",
        source=expectation_source,
        metadata={"confidence": "high", "reviewed_by": "expert"},
    )
    expectation = mlflow.get_assessment(
        trace_id=trace_id, assessment_id=logged_assessment.assessment_id
    )
    assert expectation.name == "expected_answer"
    assert expectation.value == "MLflow is an open-source machine learning platform"
    assert expectation.source.source_type == AssessmentSourceType.HUMAN
    assert expectation.metadata["confidence"] == "high"


def test_log_feedback_on_otel_trace(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-feedback-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "feedback-test-service")

    # Create a span representing a model prediction
    with tracer.start_as_current_span("prediction-span") as span:
        span.set_attribute("model", "gpt-4")
        span.set_attribute("prediction", "The weather is sunny")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )
    assert len(traces) > 0, "No traces found in the database"
    trace_id = traces[0].info.trace_id

    llm_source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4o-mini"
    )

    logged_quality = mlflow.log_feedback(
        trace_id=trace_id,
        name="quality_score",
        value=8.5,
        source=llm_source,
        metadata={"scale": "1-10", "criterion": "accuracy"},
    )
    feedback = mlflow.get_assessment(trace_id=trace_id, assessment_id=logged_quality.assessment_id)
    assert feedback.name == "quality_score"
    assert feedback.value == 8.5
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE

    human_source = AssessmentSource(
        source_type=AssessmentSourceType.HUMAN, source_id="reviewer@example.com"
    )

    logged_approval = mlflow.log_feedback(
        trace_id=trace_id,
        name="approved",
        value=True,
        source=human_source,
        metadata={"review_date": "2024-01-15"},
    )
    feedback = mlflow.get_assessment(trace_id=trace_id, assessment_id=logged_approval.assessment_id)
    assert feedback.name == "approved"
    assert feedback.value is True
    assert feedback.source.source_type == AssessmentSourceType.HUMAN


def test_multiple_assessments_on_otel_trace(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("otel-multi-assessment-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "multi-assessment-test-service")

    # Create a complex trace with nested spans
    with tracer.start_as_current_span("conversation") as parent_span:
        parent_span.set_attribute("user_query", "Explain quantum computing")

        with tracer.start_as_current_span("retrieval") as retrieval_span:
            retrieval_span.set_attribute("documents_found", 5)

        with tracer.start_as_current_span("generation") as generation_span:
            generation_span.set_attribute("model", "gpt-4")
            generation_span.set_attribute("response", "Quantum computing uses quantum mechanics...")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )
    trace_id = traces[0].info.trace_id

    mlflow.set_trace_tag(trace_id, "topic", "quantum_computing")
    mlflow.set_trace_tag(trace_id, "complexity", "high")

    human_source = AssessmentSource(AssessmentSourceType.HUMAN, "expert@physics.edu")
    llm_source = AssessmentSource(AssessmentSourceType.LLM_JUDGE, "claude-3")

    expectation = Expectation(
        name="expected_quality",
        value="Should explain quantum superposition and entanglement",
        source=human_source,
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=expectation)
    feedback_items = [
        Feedback(name="accuracy", value=9.0, source=llm_source, metadata={"max_score": "10"}),
        Feedback(name="clarity", value=8.5, source=llm_source, metadata={"max_score": "10"}),
        Feedback(
            name="helpfulness",
            value=True,
            source=human_source,
            metadata={"reviewer_expertise": "quantum_physics"},
        ),
        Feedback(
            name="contains_errors",
            value=False,
            source=human_source,
            metadata={"fact_checked": "True"},
        ),
    ]

    for feedback in feedback_items:
        mlflow.log_assessment(trace_id=trace_id, assessment=feedback)

    retrieved_trace = mlflow.get_trace(trace_id)
    assessments = retrieved_trace.info.assessments
    assert len(assessments) == 5
    assert [a.name for a in assessments] == [
        "expected_quality",
        "accuracy",
        "clarity",
        "helpfulness",
        "contains_errors",
    ]

    assert retrieved_trace.info.tags["topic"] == "quantum_computing"
    assert retrieved_trace.info.tags["complexity"] == "high"

    assert len(retrieved_trace.data.spans) == 3
    span_names = {span.name for span in retrieved_trace.data.spans}
    assert span_names == {"conversation", "retrieval", "generation"}

    tagged_traces = mlflow.search_traces(
        locations=[experiment_id],
        filter_string='tags.topic = "quantum_computing"',
        return_type="list",
    )
    assert len(tagged_traces) == 1
    assert tagged_traces[0].info.trace_id == trace_id


def test_span_kind_translation(mlflow_server: str, is_async):
    experiment = mlflow.set_experiment("span-kind-translation-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "span-kind-translation-test-service")

    with tracer.start_as_current_span("llm-call") as span:
        span.set_attribute(OpenInferenceTranslator.SPAN_KIND_ATTRIBUTE_KEY, "LLM")

    with tracer.start_as_current_span("retriever-call") as span:
        span.set_attribute(OpenInferenceTranslator.SPAN_KIND_ATTRIBUTE_KEY, "RETRIEVER")

    with tracer.start_as_current_span("tool-call") as span:
        span.set_attribute(TraceloopTranslator.SPAN_KIND_ATTRIBUTE_KEY, "tool")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )

    assert len(traces) == 3
    for trace_info in traces:
        retrieved_trace = mlflow.get_trace(trace_info.info.trace_id)
        for span in retrieved_trace.data.spans:
            if span.name == "llm-call":
                assert span.span_type == "LLM"
            elif span.name == "retriever-call":
                assert span.span_type == "RETRIEVER"
            elif span.name == "tool-call":
                assert span.span_type == "TOOL"


@pytest.mark.parametrize(
    "translator", [GenAiTranslator, OpenInferenceTranslator, TraceloopTranslator]
)
def test_span_inputs_outputs_translation(
    mlflow_server: str, is_async, translator: OtelSchemaTranslator
):
    experiment = mlflow.set_experiment("span-inputs-outputs-translation-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(
        mlflow_server, experiment_id, "span-inputs-outputs-translation-test-service"
    )

    with tracer.start_as_current_span("llm-call") as span:
        span.set_attribute(translator.INPUT_VALUE_KEYS[0], "Hello, world!")
        span.set_attribute(translator.OUTPUT_VALUE_KEYS[0], "Bye!")

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )
    assert len(traces) == 1
    retrieved_trace = mlflow.get_trace(traces[0].info.trace_id)
    assert retrieved_trace.data.spans[0].inputs == "Hello, world!"
    assert retrieved_trace.data.spans[0].outputs == "Bye!"
    assert retrieved_trace.info.request_preview == '"Hello, world!"'
    assert retrieved_trace.info.response_preview == '"Bye!"'


@pytest.mark.parametrize(
    "translator", [GenAiTranslator, OpenInferenceTranslator, TraceloopTranslator]
)
def test_span_token_usage_translation(
    mlflow_server: str, is_async, translator: OtelSchemaTranslator
):
    experiment = mlflow.set_experiment("span-token-usage-translation-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(
        mlflow_server, experiment_id, "span-token-usage-translation-test-service"
    )

    with tracer.start_as_current_span("llm-call") as span:
        span.set_attribute(translator.INPUT_TOKEN_KEY, 100)
        span.set_attribute(translator.OUTPUT_TOKEN_KEY, 50)

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )
    assert len(traces) > 0
    for trace_info in traces:
        assert trace_info.info.token_usage == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        retrieved_trace = mlflow.get_trace(trace_info.info.trace_id)
        assert (
            retrieved_trace.data.spans[0].attributes[SpanAttributeKey.CHAT_USAGE]
            == trace_info.info.token_usage
        )


@pytest.mark.parametrize(
    "translator", [GenAiTranslator, OpenInferenceTranslator, TraceloopTranslator]
)
def test_aggregated_token_usage_from_multiple_spans(
    mlflow_server: str, is_async, translator: OtelSchemaTranslator
):
    experiment = mlflow.set_experiment("aggregated-token-usage-test")
    experiment_id = experiment.experiment_id

    tracer = create_tracer(mlflow_server, experiment_id, "token-aggregation-service")

    with tracer.start_as_current_span("parent-llm-call") as parent:
        parent.set_attribute(translator.INPUT_TOKEN_KEY, 100)
        parent.set_attribute(translator.OUTPUT_TOKEN_KEY, 50)

        with tracer.start_as_current_span("child-llm-call-1") as child1:
            child1.set_attribute(translator.INPUT_TOKEN_KEY, 200)
            child1.set_attribute(translator.OUTPUT_TOKEN_KEY, 75)

        with tracer.start_as_current_span("child-llm-call-2") as child2:
            child2.set_attribute(translator.INPUT_TOKEN_KEY, 150)
            child2.set_attribute(translator.OUTPUT_TOKEN_KEY, 100)

    if is_async:
        _flush_async_logging()

    traces = mlflow.search_traces(
        locations=[experiment_id], include_spans=False, return_type="list"
    )

    trace_id = traces[0].info.trace_id
    retrieved_trace = mlflow.get_trace(trace_id)

    assert retrieved_trace.info.token_usage is not None
    assert retrieved_trace.info.token_usage["input_tokens"] == 450
    assert retrieved_trace.info.token_usage["output_tokens"] == 225
    assert retrieved_trace.info.token_usage["total_tokens"] == 675
