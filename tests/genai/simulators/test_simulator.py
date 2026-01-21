from unittest.mock import Mock, patch

import pandas as pd
import pytest

import mlflow
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.genai.simulators import ConversationSimulator
from mlflow.genai.simulators.prompts import DEFAULT_PERSONA
from mlflow.genai.simulators.simulator import _MAX_METADATA_LENGTH, SimulatedUserAgent
from mlflow.tracing.constant import TraceMetadataKey


def create_mock_evaluation_dataset(inputs: list[dict[str, object]]) -> Mock:
    mock_dataset = Mock(spec=EvaluationDataset)
    mock_dataset.to_df.return_value = pd.DataFrame({"inputs": inputs})
    return mock_dataset


def test_simulated_user_agent_generate_initial_message():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "Hello, I have a question about ML."

        agent = SimulatedUserAgent(
            goal="Learn about MLflow",
            persona="You are a beginner who asks curious questions.",
        )

        message = agent.generate_message([], turn=0)

        assert message == "Hello, I have a question about ML."
        mock_invoke.assert_called_once()

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "Learn about MLflow" in prompt
        assert "beginner" in prompt


def test_simulated_user_agent_generate_followup_message():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "Can you tell me more?"

        agent = SimulatedUserAgent(
            goal="Learn about MLflow",
        )

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        message = agent.generate_message(history, turn=1)

        assert message == "Can you tell me more?"
        mock_invoke.assert_called_once()

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "Hi there!" in prompt


def test_simulated_user_agent_default_persona():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "Test message"

        agent = SimulatedUserAgent(
            goal="Learn about ML",
        )

        message = agent.generate_message([], turn=0)

        assert message == "Test message"

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "inquisitive user" in prompt.lower()


def test_conversation_simulator_basic_simulation(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace") as mock_update_current_trace,
    ):
        # Each turn: generate_message + _check_goal_achieved
        mock_invoke.side_effect = [
            "What is MLflow?",  # turn 0 generate_message
            '{"rationale": "Goal not achieved yet", "result": "no"}',  # turn 0 goal check
            "Can you explain more?",  # turn 1 generate_message
            '{"rationale": "Goal not achieved yet", "result": "no"}',  # turn 1 goal check
        ]

        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=2,
        )

        all_trace_ids = simulator._simulate(mock_predict_fn)

        assert len(all_trace_ids) == 1  # 1 test case
        assert len(all_trace_ids[0]) == 2  # 2 trace IDs
        assert all_trace_ids[0] == ["trace_123", "trace_123"]
        assert mock_invoke.call_count == 4  # 2 turns * 2 calls each
        assert mock_update_current_trace.call_count == 2


def test_conversation_simulator_max_turns_stopping(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
    ):
        mock_invoke.side_effect = [
            "Test message",  # turn 0 generate_message
            '{"rationale": "Not yet", "result": "no"}',  # turn 0 goal check
            "Test message",  # turn 1 generate_message
            '{"rationale": "Not yet", "result": "no"}',  # turn 1 goal check
            "Test message",  # turn 2 generate_message
            '{"rationale": "Not yet", "result": "no"}',  # turn 2 goal check
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=3,
        )

        all_trace_ids = simulator._simulate(mock_predict_fn)

        # all_trace_ids is list of lists: one list per test case
        assert len(all_trace_ids) == 1  # 1 test case
        assert len(all_trace_ids[0]) == 3  # 3 trace IDs
        assert mock_invoke.call_count == 6  # 3 turns * 2 calls each


def test_conversation_simulator_empty_response_stopping(simple_test_case):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
    ):
        mock_invoke.return_value = "Test message"
        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        def empty_predict_fn(input=None, messages=None):
            return {
                "output": [
                    {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ""}],
                    }
                ]
            }

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=5,
        )

        all_trace_ids = simulator._simulate(empty_predict_fn)

        assert len(all_trace_ids) == 1
        assert len(all_trace_ids[0]) == 1  # Only 1 trace ID before stopping
        # Only generate_message called, goal check not called due to empty response
        assert mock_invoke.call_count == 1


def test_conversation_simulator_goal_achieved_stopping(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
    ):
        mock_invoke.side_effect = [
            "Test message",  # turn 0 generate_message
            '{"rationale": "Goal achieved!", "result": "yes"}',  # turn 0 goal check -> stop
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=5,
        )

        all_trace_ids = simulator._simulate(mock_predict_fn)

        assert len(all_trace_ids) == 1
        # Only 1 trace ID before goal was achieved
        assert len(all_trace_ids[0]) == 1
        # 2 calls: generate_message + goal check
        assert mock_invoke.call_count == 2
        # Verify goal check was the second call with goal check prompt
        goal_check_call = mock_invoke.call_args_list[1]
        goal_check_prompt = goal_check_call.kwargs["messages"][0].content
        assert "achieved" in goal_check_prompt.lower()


def test_conversation_simulator_context_passing(test_case_with_context):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
    ):
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Not achieved", "result": "no"}',
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        captured_kwargs = {}

        def capturing_predict_fn(input=None, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "output": [
                    {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Response"}],
                    }
                ]
            }

        simulator = ConversationSimulator(
            test_cases=[test_case_with_context],
            max_turns=1,
        )

        all_trace_ids = simulator._simulate(capturing_predict_fn)

        assert len(all_trace_ids) == 1
        assert len(all_trace_ids[0]) == 1
        # Verify context was passed to predict_fn
        assert captured_kwargs.get("user_id") == "U001"
        assert captured_kwargs.get("session_id") == "S001"


def test_conversation_simulator_multiple_test_cases(
    simple_test_case, test_case_with_persona, mock_predict_fn
):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
    ):
        # 2 test cases * 2 turns each * 2 calls per turn = 8 calls
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Not yet", "result": "no"}',
            "Test message",
            '{"rationale": "Not yet", "result": "no"}',
            "Test message",
            '{"rationale": "Not yet", "result": "no"}',
            "Test message",
            '{"rationale": "Not yet", "result": "no"}',
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case, test_case_with_persona],
            max_turns=2,
        )

        all_trace_ids = simulator._simulate(mock_predict_fn)

        assert len(all_trace_ids) == 2  # 2 test cases
        assert len(all_trace_ids[0]) == 2  # 2 trace IDs for first test case
        assert len(all_trace_ids[1]) == 2  # 2 trace IDs for second test case


@pytest.mark.parametrize(
    ("test_cases", "expected_error"),
    [
        ([], "test_cases cannot be empty"),
        ([{"persona": "test"}], r"indices \[0\].*'goal' field"),
        (
            [{"goal": "valid"}, {"persona": "missing goal"}],
            r"indices \[1\].*'goal' field",
        ),
        (
            [{"persona": "a"}, {"goal": "valid"}, {"persona": "b"}],
            r"indices \[0, 2\].*'goal' field",
        ),
    ],
    ids=[
        "empty_test_cases",
        "missing_goal",
        "second_case_missing_goal",
        "multiple_missing_goals",
    ],
)
def test_conversation_simulator_validation(test_cases, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        ConversationSimulator(
            test_cases=test_cases,
            max_turns=2,
        )


@pytest.mark.parametrize(
    "inputs",
    [
        [{"goal": "Learn about MLflow"}],
        [{"goal": "Debug issue", "persona": "Engineer"}],
        [{"goal": "Ask questions", "persona": "Student", "context": {"id": "1"}}],
    ],
)
def test_conversation_simulator_evaluation_dataset_valid(inputs):
    mock_dataset = create_mock_evaluation_dataset(inputs)
    simulator = ConversationSimulator(test_cases=mock_dataset, max_turns=2)
    assert len(simulator.test_cases) == len(inputs)
    assert simulator.test_cases == inputs


@pytest.mark.parametrize(
    "inputs",
    [
        [{"request": "What is MLflow?"}],
        [{"inputs": {"query": "Help me"}, "expected_response": "Sure!"}],
        [{"inputs": {"question": "How to log?", "answer": "Use mlflow.log"}}],
        [],
    ],
)
def test_conversation_simulator_evaluation_dataset_invalid(inputs):
    mock_dataset = create_mock_evaluation_dataset(inputs)
    with pytest.raises(ValueError, match="conversational test cases with a 'goal' field"):
        ConversationSimulator(test_cases=mock_dataset, max_turns=2)


def test_reassignment_with_valid_test_cases(simple_test_case):
    simulator = ConversationSimulator(test_cases=[simple_test_case], max_turns=2)
    new_test_cases = [
        {"goal": "New goal"},
    ]
    simulator.test_cases = new_test_cases
    assert simulator.test_cases == new_test_cases
    assert len(simulator.test_cases) == 1


def test_reassignment_with_dataframe(simple_test_case):
    simulator = ConversationSimulator(test_cases=[simple_test_case], max_turns=2)
    df = pd.DataFrame([{"goal": "Goal from DataFrame", "persona": "Analyst"}])
    simulator.test_cases = df
    assert simulator.test_cases == [{"goal": "Goal from DataFrame", "persona": "Analyst"}]


@pytest.mark.parametrize(
    ("invalid_test_cases", "expected_error"),
    [
        ([], "test_cases cannot be empty"),
        ([{"persona": "no goal here"}], r"indices \[0\].*'goal' field"),
    ],
)
def test_reassignment_with_invalid_test_cases_raises_error(
    simple_test_case, invalid_test_cases, expected_error
):
    simulator = ConversationSimulator(test_cases=[simple_test_case], max_turns=2)
    original_test_cases = simulator.test_cases
    with pytest.raises(ValueError, match=expected_error):
        simulator.test_cases = invalid_test_cases
    assert simulator.test_cases == original_test_cases


def test_conversation_simulator_sets_span_attributes(mock_predict_fn_with_context):
    long_goal = "A" * 500
    long_persona = "B" * 500
    context = {"user_id": "U001", "session_id": "S001"}

    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Goal achieved!", "result": "yes"}',
        ]

        simulator = ConversationSimulator(
            test_cases=[{"goal": long_goal, "persona": long_persona, "context": context}],
            max_turns=1,
        )

        trace_ids = simulator._simulate(mock_predict_fn_with_context)

        trace = mlflow.get_trace(trace_ids[0][0])
        root_span = trace.data.spans[0]
        metadata = trace.info.request_metadata

        assert root_span.attributes["mlflow.simulation.goal"] == long_goal
        assert root_span.attributes["mlflow.simulation.persona"] == long_persona
        assert root_span.attributes["mlflow.simulation.context"] == context
        assert metadata["mlflow.simulation.goal"] == long_goal[:_MAX_METADATA_LENGTH]
        assert metadata["mlflow.simulation.persona"] == long_persona[:_MAX_METADATA_LENGTH]


def test_conversation_simulator_uses_default_persona_and_empty_context(mock_predict_fn):
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Goal achieved!", "result": "yes"}',
        ]

        simulator = ConversationSimulator(
            test_cases=[{"goal": "Test goal"}],
            max_turns=1,
        )

        trace_ids = simulator._simulate(mock_predict_fn)

        trace = mlflow.get_trace(trace_ids[0][0])
        root_span = trace.data.spans[0]

        assert root_span.attributes["mlflow.simulation.goal"] == "Test goal"
        assert root_span.attributes["mlflow.simulation.persona"] == DEFAULT_PERSONA
        assert root_span.attributes["mlflow.simulation.context"] == {}


def test_conversation_simulator_logs_expectations_to_first_trace(mock_predict_fn):
    expectations = {"expected_topic": "MLflow", "expected_sentiment": "positive"}

    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Not achieved", "result": "no"}',
            "Follow up message",
            '{"rationale": "Goal achieved!", "result": "yes"}',
        ]

        simulator = ConversationSimulator(
            test_cases=[{"goal": "Test goal", "expectations": expectations}],
            max_turns=2,
        )

        trace_ids = simulator._simulate(mock_predict_fn)

        assert len(trace_ids[0]) == 2

        first_trace = mlflow.get_trace(trace_ids[0][0])
        first_trace_assessments = first_trace.info.assessments
        expectation_assessments = [a for a in first_trace_assessments if a.expectation is not None]

        assert len(expectation_assessments) == 2
        exp_names = {a.name for a in expectation_assessments}
        assert exp_names == {"expected_topic", "expected_sentiment"}

        for assessment in expectation_assessments:
            assert TraceMetadataKey.TRACE_SESSION in assessment.metadata

        second_trace = mlflow.get_trace(trace_ids[0][1])
        second_trace_assessments = second_trace.info.assessments
        second_expectation_assessments = [
            a for a in second_trace_assessments if a.expectation is not None
        ]
        assert len(second_expectation_assessments) == 0
