from unittest.mock import MagicMock, patch

import pytest

from mlflow.genai.simulators import ConversationSimulator
from mlflow.genai.simulators.simulator import SimulatedUserAgent


def test_simulated_user_agent_generate_initial_message():
    with patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke:
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
    with patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke:
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
    with patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke:
        mock_invoke.return_value = "Test message"

        agent = SimulatedUserAgent(
            goal="Learn about ML",
        )

        message = agent.generate_message([], turn=0)

        assert message == "Test message"

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "helpful user" in prompt.lower()


def test_conversation_simulator_basic_simulation(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace") as mock_update_current_trace,
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        # Each turn: generate_message + _check_goal_achieved
        mock_invoke.side_effect = [
            "What is MLflow?",  # turn 0 generate_message
            "no",  # turn 0 goal check
            "Can you explain more?",  # turn 1 generate_message
            "no",  # turn 1 goal check
        ]

        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=2,
        )

        all_traces = simulator._simulate(mock_predict_fn)

        assert len(all_traces) == 1  # 1 test case
        assert len(all_traces[0]) == 2  # 2 turns (traces)
        assert mock_invoke.call_count == 4  # 2 turns * 2 calls each
        assert mock_update_current_trace.call_count == 2


def test_conversation_simulator_max_turns_stopping(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            "Test message",  # turn 0 generate_message
            "no",  # turn 0 goal check
            "Test message",  # turn 1 generate_message
            "no",  # turn 1 goal check
            "Test message",  # turn 2 generate_message
            "no",  # turn 2 goal check
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=3,
        )

        all_traces = simulator._simulate(mock_predict_fn)

        # all_traces is list of lists: one list per test case
        assert len(all_traces) == 1  # 1 test case
        assert len(all_traces[0]) == 3  # 3 turns completed
        assert mock_invoke.call_count == 6  # 3 turns * 2 calls each


def test_conversation_simulator_empty_response_stopping(simple_test_case):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.return_value = "Test message"
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
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

        all_traces = simulator._simulate(empty_predict_fn)

        assert len(all_traces) == 1
        assert len(all_traces[0]) == 1  # Only 1 turn before stopping
        # Only generate_message called, goal check not called due to empty response
        assert mock_invoke.call_count == 1


def test_conversation_simulator_goal_achieved_stopping(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            "Test message",  # turn 0 generate_message
            "yes it is achieved",  # turn 0 goal check returns "yes" -> stop
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=5,
        )

        all_traces = simulator._simulate(mock_predict_fn)

        assert len(all_traces) == 1
        # Only 1 turn completed before goal was achieved
        assert len(all_traces[0]) == 1
        # 2 calls: generate_message + goal check
        assert mock_invoke.call_count == 2
        # Verify goal check was the second call with goal check prompt
        goal_check_call = mock_invoke.call_args_list[1]
        goal_check_prompt = goal_check_call.kwargs["messages"][0].content
        assert "achieved" in goal_check_prompt.lower()
        assert "yes" in goal_check_prompt.lower() or "no" in goal_check_prompt.lower()


def test_conversation_simulator_context_passing(test_case_with_context):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            "Test message",
            "no",
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
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

        all_traces = simulator._simulate(capturing_predict_fn)

        assert len(all_traces) == 1
        assert len(all_traces[0]) == 1
        # Verify context was passed to predict_fn
        assert captured_kwargs.get("user_id") == "U001"
        assert captured_kwargs.get("session_id") == "S001"


def test_conversation_simulator_multiple_test_cases(
    simple_test_case, test_case_with_persona, mock_predict_fn
):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_model") as mock_invoke,
        patch("mlflow.trace") as mock_trace_decorator,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        # 2 test cases * 2 turns each * 2 calls per turn = 8 calls
        mock_invoke.side_effect = [
            "Test message",
            "no",
            "Test message",
            "no",
            "Test message",
            "no",
            "Test message",
            "no",
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_trace_decorator.return_value = lambda fn: fn

        simulator = ConversationSimulator(
            test_cases=[simple_test_case, test_case_with_persona],
            max_turns=2,
        )

        all_traces = simulator._simulate(mock_predict_fn)

        assert len(all_traces) == 2  # 2 test cases
        assert len(all_traces[0]) == 2  # 2 turns for first test case
        assert len(all_traces[1]) == 2  # 2 turns for second test case


@pytest.mark.parametrize(
    ("test_cases", "expected_error"),
    [
        ([], "test_cases cannot be empty"),
        ([{"persona": "test"}], "must have 'goal' field"),
        ([{"goal": "valid"}, {"persona": "missing goal"}], "must have 'goal' field"),
    ],
    ids=["empty_test_cases", "missing_goal", "second_case_missing_goal"],
)
def test_conversation_simulator_validation(test_cases, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        ConversationSimulator(
            test_cases=test_cases,
            max_turns=2,
        )
