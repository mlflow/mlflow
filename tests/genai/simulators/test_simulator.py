from unittest.mock import MagicMock, patch

import pytest

from mlflow.genai.simulators import ConversationSimulator
from mlflow.genai.simulators.simulator import SimulatedUserAgent


def test_simulated_user_agent_generate_initial_message():
    with patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke:
        mock_invoke.return_value = ("Hello, I have a question about ML.", None)

        agent = SimulatedUserAgent(
            goal="Learn about MLflow",
            persona="You are a beginner who asks curious questions.",
            model="openai/gpt-4o-mini",
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
    with patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke:
        mock_invoke.return_value = ("Can you tell me more?", None)

        agent = SimulatedUserAgent(
            goal="Learn about MLflow",
            model="openai/gpt-4o-mini",
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
    with patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke:
        mock_invoke.return_value = ("Test message", None)

        agent = SimulatedUserAgent(
            goal="Learn about ML",
            model="openai/gpt-4o-mini",
        )

        message = agent.generate_message([], turn=0)

        assert message == "Test message"

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "helpful user" in prompt.lower()


def test_conversation_simulator_basic_simulation(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke,
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace") as mock_update_current_trace,
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            ("What is MLflow?", None),
            ("no", None),
            ("Can you explain more?", None),
            ("no", None),
        ]

        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_start_span.return_value.__enter__.return_value = MagicMock()

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=2,
            user_model="openai/gpt-4o-mini",
        )

        traces = simulator.simulate(mock_predict_fn)

        assert len(traces) == 2
        assert mock_invoke.call_count == 4
        assert mock_update_current_trace.call_count == 2


def test_conversation_simulator_max_turns_stopping(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke,
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            ("Test message", None),
            ("no", None),
            ("Test message", None),
            ("no", None),
            ("Test message", None),
            ("no", None),
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_start_span.return_value.__enter__.return_value = MagicMock()

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=3,
        )

        traces = simulator.simulate(mock_predict_fn)

        assert len(traces) == 3


def test_conversation_simulator_empty_response_stopping(simple_test_case):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke,
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.return_value = ("Test message", None)
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_start_span.return_value.__enter__.return_value = MagicMock()

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

        traces = simulator.simulate(empty_predict_fn)

        assert len(traces) == 1
        assert mock_invoke.call_count == 1


def test_conversation_simulator_goal_achieved_stopping(simple_test_case, mock_predict_fn):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke,
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            ("Test message", None),
            ("yes it is achieved", None),
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_start_span.return_value.__enter__.return_value = MagicMock()

        simulator = ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=5,
        )

        traces = simulator.simulate(mock_predict_fn)

        assert len(traces) == 1
        assert mock_invoke.call_count == 2


def test_conversation_simulator_context_passing(
    test_case_with_context, mock_predict_fn_with_context
):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke,
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            ("Test message", None),
            ("no", None),
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_start_span.return_value.__enter__.return_value = MagicMock()

        simulator = ConversationSimulator(
            test_cases=[test_case_with_context],
            max_turns=1,
        )

        traces = simulator.simulate(mock_predict_fn_with_context)

        assert len(traces) == 1


def test_conversation_simulator_multiple_test_cases(
    simple_test_case, test_case_with_persona, mock_predict_fn
):
    with (
        patch("mlflow.genai.simulators.simulator._invoke_litellm_and_handle_tools") as mock_invoke,
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.get_last_active_trace_id") as mock_get_trace_id,
        patch("mlflow.update_current_trace"),
        patch("mlflow.get_trace") as mock_get_trace,
    ):
        mock_invoke.side_effect = [
            ("Test message", None),
            ("no", None),
            ("Test message", None),
            ("no", None),
            ("Test message", None),
            ("no", None),
            ("Test message", None),
            ("no", None),
        ]
        mock_get_trace_id.return_value = "trace_123"
        mock_trace = MagicMock()
        mock_get_trace.return_value = mock_trace
        mock_start_span.return_value.__enter__.return_value = MagicMock()

        simulator = ConversationSimulator(
            test_cases=[simple_test_case, test_case_with_persona],
            max_turns=2,
        )

        traces = simulator.simulate(mock_predict_fn)

        assert len(traces) == 4


def test_conversation_simulator_empty_test_cases(mock_predict_fn):
    simulator = ConversationSimulator(
        test_cases=[],
        max_turns=2,
    )
    with pytest.raises(ValueError, match="test_cases cannot be empty"):
        simulator.simulate(mock_predict_fn)
