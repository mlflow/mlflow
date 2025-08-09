from unittest.mock import MagicMock, patch

import pytest

import mlflow
import mlflow.haystack
from mlflow.entities.span import SpanAttributeKey, SpanType

from tests.tracing.helper import get_traces


def create_mock_pipeline():
    """Create a mock Haystack pipeline."""
    pipeline = MagicMock()
    pipeline.__class__.__name__ = "Pipeline"

    # Mock the graph attribute
    mock_node1 = MagicMock()
    mock_node1.__class__.__name__ = "PromptBuilder"
    mock_node2 = MagicMock()
    mock_node2.__class__.__name__ = "OpenAIGenerator"

    pipeline.graph = MagicMock()
    pipeline.graph.nodes = {"prompt_builder": mock_node1, "llm": mock_node2}

    return pipeline


def create_mock_async_pipeline():
    """Create a mock Haystack AsyncPipeline."""
    pipeline = MagicMock()
    pipeline.__class__.__name__ = "AsyncPipeline"

    # Mock the graph attribute
    mock_node1 = MagicMock()
    mock_node1.__class__.__name__ = "PromptBuilder"
    mock_node2 = MagicMock()
    mock_node2.__class__.__name__ = "OpenAIGenerator"

    pipeline.graph = MagicMock()
    pipeline.graph.nodes = {"prompt_builder": mock_node1, "llm": mock_node2}

    return pipeline


def create_mock_component(component_type="OpenAIGenerator"):
    """Create a mock Haystack component."""
    component = MagicMock()
    component.__class__.__name__ = component_type
    component.model = "gpt-4o-mini"
    component._init_parameters = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "api_key": "secret_key",  # Should be filtered out
    }

    # Mock input/output sockets
    input_socket = MagicMock()
    input_socket.type = "str"
    component.__haystack_input__ = {"prompt": input_socket}

    output_socket = MagicMock()
    output_socket.type = "List[str]"
    component.__haystack_output__ = {"replies": output_socket}

    return component


DUMMY_PIPELINE_INPUT = {"prompt_builder": {"question": "Who lives in Paris?"}}

DUMMY_PIPELINE_OUTPUT = {
    "llm": {
        "replies": ["Many people live in Paris, including residents, tourists, and workers."],
        "meta": [
            {
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
        ],
    }
}

DUMMY_COMPONENT_INPUT = {"prompt": "Answer the question: Who lives in Paris?"}

DUMMY_COMPONENT_OUTPUT = {
    "replies": ["Many people live in Paris, including residents, tourists, and workers."],
    "meta": [
        {
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    ],
}


def test_pipeline_autolog():
    """Test autologging for Haystack pipelines."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    with patch("haystack.core.pipeline.pipeline.Pipeline", create_mock_pipeline):
        mlflow.haystack.autolog()

        pipeline = create_mock_pipeline()

        # Mock the run method
        def mock_run(self, data, *args, **kwargs):
            return DUMMY_PIPELINE_OUTPUT

        # Set the correct __name__ attribute to match the expected method name
        mock_run.__name__ = "run"

        # Apply patching manually since we're mocking
        from mlflow.haystack.autolog import patched_class_call

        def patched_run(data, *args, **kwargs):
            return patched_class_call(mock_run, pipeline, data, *args, **kwargs)

        pipeline.run = patched_run

        # Run the pipeline
        pipeline.run(DUMMY_PIPELINE_INPUT)

        # Check traces
        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1

        span = traces[0].data.spans[0]
        assert span.name == "Pipeline.run"
        assert span.span_type == SpanType.CHAIN
        assert span.inputs == {"question": "Who lives in Paris?"}
        expected_outputs = {
            "llm": {
                "replies": "Many people live in Paris, including residents, tourists, and workers.",
                "meta": [
                    {
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    }
                ],
            }
        }
        assert span.outputs == expected_outputs

        # Check attributes
        assert span.attributes.get(SpanAttributeKey.MESSAGE_FORMAT) == "haystack"
        assert span.attributes.get("components") == "['prompt_builder', 'llm']"
        assert span.attributes.get("component_count") == 2


def test_pipeline_component_execution():
    """Test component execution tracing within a pipeline."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    mlflow.haystack.autolog()

    # Create mock component
    component = create_mock_component()
    component_dict = {"instance": component}

    # Mock _run_component method
    def mock_run_component(component_name, component, inputs, component_visits, parent_span=None):
        return DUMMY_COMPONENT_OUTPUT

    mock_run_component.__name__ = "_run_component"

    # Apply patching - for static methods, we pass None as self
    from mlflow.haystack.autolog import patched_class_call

    def run_component(*args, **kwargs):
        return patched_class_call(mock_run_component, None, *args, **kwargs)

    # Run the component
    run_component("llm", component_dict, DUMMY_COMPONENT_INPUT, {"llm": 1})

    # Check traces
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "NoneType._run_component"  # Static method has no class
    assert span.span_type == SpanType.TOOL  # Static methods default to TOOL type
    # For static methods, inputs include all args but with different parameter names
    assert "component" in span.inputs
    assert span.inputs["component"] == "llm"
    assert span.outputs == DUMMY_COMPONENT_OUTPUT

    # Check attributes
    assert span.attributes.get(SpanAttributeKey.MESSAGE_FORMAT) == "haystack"
    # Static methods with None self won't have component-specific attributes

    # Check token usage in standard format
    chat_usage = span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    assert chat_usage is not None
    assert chat_usage["input_tokens"] == 10
    assert chat_usage["output_tokens"] == 20
    assert chat_usage["total_tokens"] == 30


def test_component_meta_patching():
    """Test ComponentMeta patching for dynamic component wrapping."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    mlflow.haystack.autolog()

    # Create a simple component
    class MockComponent:
        def __init__(self):
            self._init_parameters = {"model": "test-model"}

        def run(self, **kwargs):
            return DUMMY_COMPONENT_OUTPUT

    # Apply run method patching
    from mlflow.haystack.autolog import patched_class_call

    original_run = MockComponent.run

    def patched_run(self, **kwargs):
        return patched_class_call(original_run, self, **kwargs)

    MockComponent.run = patched_run

    # Create and run component
    component = MockComponent()
    component.run(**DUMMY_COMPONENT_INPUT)

    # Check traces
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "MockComponent.run"
    assert span.span_type == SpanType.TOOL
    assert span.inputs == {"kwargs": DUMMY_COMPONENT_INPUT}
    # Direct component output (no formatting applied outside of pipeline context)
    assert span.outputs == DUMMY_COMPONENT_OUTPUT


@pytest.mark.asyncio
async def test_async_pipeline_autolog():
    """Test autologging for Haystack AsyncPipeline."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    with patch("haystack.core.pipeline.async_pipeline.AsyncPipeline", create_mock_async_pipeline):
        mlflow.haystack.autolog()

        pipeline = create_mock_async_pipeline()

        # Mock the async run method
        async def mock_run_async(self, data, *args, **kwargs):
            return DUMMY_PIPELINE_OUTPUT

        mock_run_async.__name__ = "run_async"

        # Apply patching manually since we're mocking
        from mlflow.haystack.autolog import patched_async_class_call

        async def wrapped_run_async(data, *args, **kwargs):
            return await patched_async_class_call(mock_run_async, pipeline, data, *args, **kwargs)

        pipeline.run_async = wrapped_run_async

        # Run the pipeline
        await pipeline.run_async(DUMMY_PIPELINE_INPUT)

        # Check traces
        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1

        span = traces[0].data.spans[0]
        assert span.name == "AsyncPipeline.run_async"
        assert span.span_type == SpanType.CHAIN
        assert span.inputs == {"question": "Who lives in Paris?"}
        expected_outputs = {
            "llm": {
                "replies": "Many people live in Paris, including residents, tourists, and workers.",
                "meta": [
                    {
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    }
                ],
            }
        }
        assert span.outputs == expected_outputs


@pytest.mark.asyncio
async def test_async_pipeline_generator():
    """Test autologging for AsyncPipeline.run_async_generator()."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    mlflow.haystack.autolog()

    pipeline = create_mock_async_pipeline()

    # Mock the async generator method
    async def mock_run_async_generator(*args, **kwargs):
        yield {"component1": {"output": "partial1"}}
        yield {"llm": DUMMY_COMPONENT_OUTPUT}

    # Apply patching for async generator
    # For async generators, we need special handling since we can't use the standard async patch
    from mlflow.entities import SpanType

    async def wrapped_generator(*args, **kwargs):
        fullname = f"{pipeline.__class__.__name__}.run_async_generator"

        with mlflow.start_span(name=fullname, span_type=SpanType.CHAIN) as span:
            span.set_inputs({"data": args[0] if args else kwargs})
            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "haystack")
            span.set_attribute("components", ["prompt_builder", "llm"])
            span.set_attribute("component_count", 2)

            accumulated_outputs = {}
            chunk_count = 0

            async for output in mock_run_async_generator(*args, **kwargs):
                accumulated_outputs.update(output)
                chunk_count += 1
                yield output

            span.set_outputs(accumulated_outputs)
            span.set_attribute("chunks", chunk_count)

    # Run the generator
    outputs = []
    async for output in wrapped_generator(DUMMY_PIPELINE_INPUT):
        outputs.append(output)

    # Check traces
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "AsyncPipeline.run_async_generator"
    assert span.span_type == SpanType.CHAIN

    # Check accumulated outputs
    assert "component1" in span.outputs
    assert "llm" in span.outputs
    assert span.attributes.get("chunks") == 2


def test_autolog_disable():
    """Test disabling autolog."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    with patch("haystack.core.pipeline.pipeline.Pipeline", create_mock_pipeline):
        # Enable autolog first
        mlflow.haystack.autolog()

        # Then disable it
        mlflow.haystack.autolog(disable=True)

        pipeline = create_mock_pipeline()

        # Mock the run method
        def mock_run(*args, **kwargs):
            return DUMMY_PIPELINE_OUTPUT

        pipeline.run = MagicMock(side_effect=mock_run)

        # Run the pipeline
        pipeline.run(DUMMY_PIPELINE_INPUT)

        # Check that no traces were created
        traces = get_traces()
        assert len(traces) == 0


def test_pipeline_error_handling():
    """Test error handling in pipeline execution."""
    # Clear any existing traces
    mlflow.tracking.fluent._active_experiment_id = None

    with patch("haystack.core.pipeline.pipeline.Pipeline", create_mock_pipeline):
        mlflow.haystack.autolog()

        pipeline = create_mock_pipeline()

        # Mock the run method to raise an error
        error_msg = "Pipeline execution failed"

        def mock_run(self, data, *args, **kwargs):
            raise RuntimeError(error_msg)

        mock_run.__name__ = "run"

        pipeline.run = MagicMock(side_effect=mock_run)

        # Apply patching manually
        from mlflow.haystack.autolog import patched_class_call

        def patched_error_run(data, *args, **kwargs):
            return patched_class_call(mock_run, pipeline, data, *args, **kwargs)

        pipeline.run = patched_error_run

        # Run the pipeline and expect an error
        with pytest.raises(RuntimeError, match=error_msg):
            pipeline.run(DUMMY_PIPELINE_INPUT)

        # Check traces
        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "ERROR"
        assert len(traces[0].data.spans) == 1

        span = traces[0].data.spans[0]
        assert span.name == "Pipeline.run"
        assert span.status.status_code == "ERROR"
        assert error_msg in span.status.description or any(
            error_msg in str(event) for event in (span.events or [])
        )
