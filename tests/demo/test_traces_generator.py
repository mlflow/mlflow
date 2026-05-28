import pytest

from mlflow import MlflowClient, get_experiment_by_name, set_experiment
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.traces import (
    _PROVIDER_TO_LLM_SPAN_NAME,
    DEMO_TRACE_TYPE_TAG,
    DEMO_VERSION_TAG,
    TracesDemoGenerator,
)
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey


@pytest.fixture
def traces_generator():
    generator = TracesDemoGenerator()
    original_version = generator.version
    yield generator
    TracesDemoGenerator.version = original_version


def test_generator_attributes():
    generator = TracesDemoGenerator()
    assert generator.name == DemoFeature.TRACES
    assert generator.version == 3


def test_data_exists_false_when_no_experiment():
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False


def test_data_exists_false_when_experiment_empty():
    set_experiment(DEMO_EXPERIMENT_NAME)
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False


def test_generate_creates_traces():
    generator = TracesDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.TRACES
    assert len(result.entity_ids) > 0
    assert "experiments" in result.navigation_url


def test_generate_creates_experiment():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"


def test_data_exists_true_after_generate():
    generator = TracesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_traces():
    generator = TracesDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False


def test_traces_have_expected_structure():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    assert len(traces) > 0

    all_span_names = set()
    for trace in traces:
        all_span_names.update(span.name for span in trace.data.spans)

    assert "rag_pipeline" in all_span_names
    assert "agent" in all_span_names
    assert "chat_agent" in all_span_names
    assert "prompt_chain" in all_span_names
    assert "render_prompt" in all_span_names
    assert "embed_query" in all_span_names
    assert "retrieve_docs" in all_span_names
    assert "chat.completions.create" in all_span_names  # OpenAI
    assert "messages.create" in all_span_names  # Anthropic
    assert "generate_content" in all_span_names  # Google


def test_traces_have_version_metadata():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    v1_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v1"]
    v2_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_VERSION_TAG) == "v2"]

    # 2 RAG + 2 agent + 6 prompt + 4 multimodal + 7 session = 21 per version
    assert len(v1_traces) == 21
    assert len(v2_traces) == 21
    assert len(traces) == 42


def test_traces_have_type_metadata():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=50)

    rag_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "rag"]
    agent_traces = [t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "agent"]
    prompt_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "prompt"
    ]
    session_traces = [
        t for t in traces if t.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "session"
    ]

    # 2 RAG per version = 4 total
    # 2 agent per version = 4 total
    # 6 prompt per version = 12 total
    # 7 session per version = 14 total
    assert len(rag_traces) == 4
    assert len(agent_traces) == 4
    assert len(prompt_traces) == 12
    assert len(session_traces) == 14


def test_is_generated_checks_version(traces_generator):
    traces_generator.generate()
    traces_generator.store_version()

    assert traces_generator.is_generated() is True

    TracesDemoGenerator.version = 99
    assert traces_generator.is_generated() is False


def _is_chat_message(obj):
    """
    chat-utils/openai.ts has a normalizeOpenAIChatInput function
    that asserts a chat-renderable message for inputs
    """
    return isinstance(obj, dict) and "role" in obj and "content" in obj


def _has_openai_choices_shape(outputs):
    """
    ModalTraceExplorer.utils.tsx has a fallback which tries to
    normalise responses if they are OpenAI-shaped
    """
    if not isinstance(outputs, dict):
        return False
    choices = outputs.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    return all(_is_chat_message(c.get("message")) for c in choices)


def test_root_span_inputs_are_chat_renderable():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    for trace in traces:
        root = next(s for s in trace.data.spans if s.parent_id is None)
        inputs = root.inputs
        assert isinstance(inputs, dict), f"Root span {root.name} inputs is not a dict"
        messages = inputs.get("messages")
        assert isinstance(messages, list), f"Root span {root.name} inputs missing 'messages' list"
        assert messages, f"Root span {root.name} inputs has empty 'messages' list"
        assert all(_is_chat_message(m) for m in messages), (
            f"Root span {root.name} has malformed message in inputs"
        )


def test_llm_span_outputs_are_chat_renderable():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    for trace in traces:
        for span in trace.data.spans:
            if span.span_type != SpanType.LLM:
                continue
            assert _has_openai_choices_shape(span.outputs), (
                f"LLM span {span.name} in trace {trace.info.trace_id} "
                f"does not have OpenAI choices output shape"
            )


def test_root_span_outputs_are_chat_renderable():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    for trace in traces:
        # Multimodal traces use the OpenAI Images / Audio API response shapes,
        # not ChatCompletions; both render in the UI but via different normalizers.
        if trace.info.trace_metadata.get(DEMO_TRACE_TYPE_TAG) == "multimodal":
            continue
        root = next(s for s in trace.data.spans if s.parent_id is None)
        assert _has_openai_choices_shape(root.outputs), (
            f"Root span {root.name} does not have OpenAI choices output shape"
        )


def test_trace_with_tools_has_react_shape():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    for trace in traces:
        root = next(s for s in trace.data.spans if s.parent_id is None)
        children = [s for s in trace.data.spans if s.parent_id == root.span_id]
        tool_count = sum(1 for s in children if s.span_type == SpanType.TOOL)
        if tool_count == 0:
            continue
        assert len(children) == 2 * tool_count + 1
        ordered = sorted(children, key=lambda s: s.start_time_ns)
        # Assert ordering now
        expected = [SpanType.LLM, SpanType.TOOL] * tool_count + [SpanType.LLM]
        assert [s.span_type for s in ordered] == expected


def test_final_llm_span_emits_content():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    for trace in traces:
        llm_spans = [s for s in trace.data.spans if s.span_type == SpanType.LLM]
        if not llm_spans:
            continue
        last_llm = max(llm_spans, key=lambda s: s.start_time_ns)
        choices = last_llm.outputs.get("choices", [])
        content = choices[0].get("message", {}).get("content") if choices else None
        assert isinstance(content, str)
        assert content


def test_span_name_matches_provider():
    generator = TracesDemoGenerator()
    generator.generate()

    experiment = get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    client = MlflowClient()
    traces = client.search_traces(locations=[experiment.experiment_id], max_results=100)

    for trace in traces:
        for span in trace.data.spans:
            # Multimodal traces are of SpanType.CHAT_MODEL and so will be caught here
            if span.span_type != SpanType.LLM:
                continue
            provider = span.attributes.get(SpanAttributeKey.MODEL_PROVIDER)
            assert span.name == _PROVIDER_TO_LLM_SPAN_NAME[provider]
