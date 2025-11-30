
import json
import pytest
from unittest import mock
from mlflow.genai import make_judge
from mlflow.entities import Trace, TraceInfo, TraceData, TraceLocation, TraceState
from mlflow.entities.assessment import Expectation, AssessmentSource, AssessmentSourceType, Feedback
import time

@pytest.fixture
def mock_invoke_judge_model(monkeypatch):
    """Unified fixture that captures all invocation details."""
    calls = []
    captured_args = {}

    def _mock(
        model_uri,
        prompt,
        assessment_name,
        trace=None,
        num_retries=10,
        response_format=None,
        use_case=None,
    ):
        calls.append((model_uri, prompt, assessment_name))
        captured_args.update(
            {
                "model_uri": model_uri,
                "prompt": prompt,
                "assessment_name": assessment_name,
                "trace": trace,
            }
        )
        return Feedback(name=assessment_name, value=True, rationale="Test passed")

    import mlflow.genai.judges.instructions_judge
    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", _mock)

    _mock.calls = calls
    _mock.captured_args = captured_args
    return _mock

def create_manual_trace(trace_id, session_id, inputs, outputs, expectations=None):
    trace_metadata = {
        "mlflow.traceInputs": json.dumps(inputs),
        "mlflow.traceOutputs": json.dumps(outputs),
        "mlflow.trace.session": session_id
    }
    
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=int(time.time() * 1000),
        execution_duration=100,
        state=TraceState.OK,
        trace_metadata=trace_metadata,
        tags={},
        assessments=expectations or []
    )
    
    trace_data = TraceData(spans=[])
    trace = Trace(info=trace_info, data=trace_data)
    return trace



def test_session_expectations_extraction(mock_invoke_judge_model):

    """Test that expectations are correctly extracted from session traces."""
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }} against {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    # Add expectation to trace1
    exp1 = Expectation(
        name="politeness",
        value="Should be polite",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )

    trace1 = create_manual_trace(
        "trace-1",
        "session-1",
        inputs={"question": "Hello"},
        outputs={"answer": "Hi"},
        expectations=[exp1]
    )

    trace2 = create_manual_trace(
        "trace-2",
        "session-1",
        inputs={"question": "How are you?"},
        outputs={"answer": "Good"},
    )

    result = judge(session=[trace1, trace2])

    assert isinstance(result, Feedback)
    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    expected_expectations_json = json.dumps(
        {"politeness": "Should be polite"}, default=str, indent=2
    )
    assert f"expectations: {expected_expectations_json}" in user_msg.content
