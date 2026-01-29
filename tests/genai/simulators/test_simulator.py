from unittest.mock import Mock, patch

import pandas as pd
import pytest

from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.genai.simulators import (
    BaseSimulatedUserAgent,
    ConversationSimulator,
    SimulatedUserAgent,
    SimulatorContext,
)
from mlflow.genai.simulators.prompts import DEFAULT_PERSONA
from mlflow.genai.simulators.simulator import _MAX_METADATA_LENGTH
from mlflow.tracing.constant import TraceMetadataKey


def create_mock_evaluation_dataset(inputs: list[dict[str, object]]) -> Mock:
    mock_dataset = Mock(spec=EvaluationDataset)
    mock_dataset.to_df.return_value = pd.DataFrame({"inputs": inputs})
    return mock_dataset


def test_simulated_user_agent_generate_initial_message():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "Hello, I have a question about ML."

        agent = SimulatedUserAgent()
        context = SimulatorContext(
            goal="Learn about MLflow",
            persona="You are a beginner who asks curious questions.",
            conversation_history=[],
            turn=0,
        )

        message = agent.generate_message(context)

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

        agent = SimulatedUserAgent()
        context = SimulatorContext(
            goal="Learn about MLflow",
            persona="A helpful user",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            turn=1,
        )

        message = agent.generate_message(context)

        assert message == "Can you tell me more?"
        mock_invoke.assert_called_once()

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "Hi there!" in prompt


def test_simulated_user_agent_default_persona():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "Test message"

        agent = SimulatedUserAgent()
        context = SimulatorContext(
            goal="Learn about ML",
            persona=DEFAULT_PERSONA,
            conversation_history=[],
            turn=0,
        )

        message = agent.generate_message(context)

        assert message == "Test message"

        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        prompt = messages[0].content

        assert "inquisitive user" in prompt.lower()


def test_conversation_simulator_basic_simulation(
    simple_test_case, mock_predict_fn, simulation_mocks
):
    # Each turn: generate_message + _check_goal_achieved
    simulation_mocks["invoke"].side_effect = [
        "What is MLflow?",  # turn 0 generate_message
        '{"rationale": "Goal not achieved yet", "result": "no"}',  # turn 0 goal check
        "Can you explain more?",  # turn 1 generate_message
        '{"rationale": "Goal not achieved yet", "result": "no"}',  # turn 1 goal check
    ]

    simulator = ConversationSimulator(
        test_cases=[simple_test_case],
        max_turns=2,
    )

    all_traces = simulator.simulate(mock_predict_fn)

    assert len(all_traces) == 1  # 1 test case
    assert len(all_traces[0]) == 2  # 2 traces
    assert all(t is simulation_mocks["trace"] for t in all_traces[0])
    assert simulation_mocks["invoke"].call_count == 4  # 2 turns * 2 calls each
    assert simulation_mocks["update_trace"].call_count == 2


def test_conversation_simulator_max_turns_stopping(
    simple_test_case, mock_predict_fn, simulation_mocks
):
    simulation_mocks["invoke"].side_effect = [
        "Test message",  # turn 0 generate_message
        '{"rationale": "Not yet", "result": "no"}',  # turn 0 goal check
        "Test message",  # turn 1 generate_message
        '{"rationale": "Not yet", "result": "no"}',  # turn 1 goal check
        "Test message",  # turn 2 generate_message
        '{"rationale": "Not yet", "result": "no"}',  # turn 2 goal check
    ]

    simulator = ConversationSimulator(
        test_cases=[simple_test_case],
        max_turns=3,
    )

    all_traces = simulator.simulate(mock_predict_fn)

    assert len(all_traces) == 1  # 1 test case
    assert len(all_traces[0]) == 3  # 3 traces
    assert simulation_mocks["invoke"].call_count == 6  # 3 turns * 2 calls each


def test_conversation_simulator_empty_response_stopping(simple_test_case, simulation_mocks):
    simulation_mocks["invoke"].return_value = "Test message"

    def empty_predict_fn(input=None, messages=None, **kwargs):
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

    all_traces = simulator.simulate(empty_predict_fn)

    assert len(all_traces) == 1
    assert len(all_traces[0]) == 1  # Only 1 trace before stopping
    # Only generate_message called, goal check not called due to empty response
    assert simulation_mocks["invoke"].call_count == 1


def test_conversation_simulator_goal_achieved_stopping(
    simple_test_case, mock_predict_fn, simulation_mocks
):
    simulation_mocks["invoke"].side_effect = [
        "Test message",  # turn 0 generate_message
        '{"rationale": "Goal achieved!", "result": "yes"}',  # turn 0 goal check -> stop
    ]

    simulator = ConversationSimulator(
        test_cases=[simple_test_case],
        max_turns=5,
    )

    all_traces = simulator.simulate(mock_predict_fn)

    assert len(all_traces) == 1
    # Only 1 trace before goal was achieved
    assert len(all_traces[0]) == 1
    # 2 calls: generate_message + goal check
    assert simulation_mocks["invoke"].call_count == 2
    # Verify goal check was the second call with goal check prompt
    goal_check_call = simulation_mocks["invoke"].call_args_list[1]
    goal_check_prompt = goal_check_call.kwargs["messages"][0].content
    assert "achieved" in goal_check_prompt.lower()


def test_conversation_simulator_context_passing(test_case_with_context, simulation_mocks):
    simulation_mocks["invoke"].side_effect = [
        "Test message",
        '{"rationale": "Not achieved", "result": "no"}',
    ]

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

    all_traces = simulator.simulate(capturing_predict_fn)

    assert len(all_traces) == 1
    assert len(all_traces[0]) == 1
    # Verify context was passed to predict_fn
    assert captured_kwargs.get("user_id") == "U001"
    assert captured_kwargs.get("session_id") == "S001"


def test_conversation_simulator_mlflow_session_id_passed_to_predict_fn(
    simple_test_case, simulation_mocks
):
    simulation_mocks["invoke"].side_effect = [
        "Test message",
        '{"rationale": "Not yet", "result": "no"}',
        "Test message 2",
        '{"rationale": "Not yet", "result": "no"}',
    ]

    captured_session_ids = []

    def capturing_predict_fn(input=None, **kwargs):
        captured_session_ids.append(kwargs.get("mlflow_session_id"))
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
        test_cases=[simple_test_case],
        max_turns=2,
    )

    all_traces = simulator.simulate(capturing_predict_fn)

    assert len(all_traces) == 1
    assert len(all_traces[0]) == 2
    # Verify mlflow_session_id was passed to predict_fn
    assert len(captured_session_ids) == 2
    assert all(sid is not None for sid in captured_session_ids)
    assert all(sid.startswith("sim-") for sid in captured_session_ids)
    # Verify session ID is consistent across all turns in the same conversation
    assert captured_session_ids[0] == captured_session_ids[1]


def test_conversation_simulator_multiple_test_cases(
    simple_test_case, test_case_with_persona, mock_predict_fn, simulation_mocks
):
    # 2 test cases * 2 turns each * 2 calls per turn = 8 calls
    simulation_mocks["invoke"].side_effect = [
        "Test message",
        '{"rationale": "Not yet", "result": "no"}',
        "Test message",
        '{"rationale": "Not yet", "result": "no"}',
        "Test message",
        '{"rationale": "Not yet", "result": "no"}',
        "Test message",
        '{"rationale": "Not yet", "result": "no"}',
    ]

    simulator = ConversationSimulator(
        test_cases=[simple_test_case, test_case_with_persona],
        max_turns=2,
    )

    all_traces = simulator.simulate(mock_predict_fn)

    assert len(all_traces) == 2  # 2 test cases
    assert len(all_traces[0]) == 2  # 2 traces for first test case
    assert len(all_traces[1]) == 2  # 2 traces for second test case


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


def test_simulator_context_is_first_turn():
    context_first = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[],
        turn=0,
    )
    assert context_first.is_first_turn is True

    context_later = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[{"role": "user", "content": "Hello"}],
        turn=1,
    )
    assert context_later.is_first_turn is False


def test_simulator_context_formatted_history():
    context_empty = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[],
        turn=0,
    )
    assert context_empty.formatted_history is None

    context_with_history = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        turn=1,
    )
    assert context_with_history.formatted_history == "user: Hello\nassistant: Hi there!"


def test_simulator_context_last_assistant_response():
    context_empty = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[],
        turn=0,
    )
    assert context_empty.last_assistant_response is None

    context_with_history = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        turn=1,
    )
    assert context_with_history.last_assistant_response == "Hi there!"


def test_simulator_context_is_frozen():
    context = SimulatorContext(
        goal="Test goal",
        persona="Test persona",
        conversation_history=[],
        turn=0,
    )
    with pytest.raises(AttributeError, match="cannot assign to field"):
        context.goal = "New goal"


def test_custom_user_agent_class(simple_test_case, mock_predict_fn, simulation_mocks):
    class CustomUserAgent(BaseSimulatedUserAgent):
        def generate_message(self, context: SimulatorContext) -> str:
            return f"Custom message for: {context.goal}"

    simulation_mocks["invoke"].return_value = '{"rationale": "Goal achieved!", "result": "yes"}'

    simulator = ConversationSimulator(
        test_cases=[simple_test_case],
        max_turns=2,
        user_agent_class=CustomUserAgent,
    )

    all_traces = simulator.simulate(mock_predict_fn)

    assert len(all_traces) == 1
    assert len(all_traces[0]) == 1


def test_user_agent_class_default(simple_test_case):
    simulator = ConversationSimulator(
        test_cases=[simple_test_case],
        max_turns=2,
    )
    assert simulator.user_agent_class is SimulatedUserAgent


def test_user_agent_class_receives_context(simple_test_case, mock_predict_fn, simulation_mocks):
    captured_contexts = []

    class ContextCapturingAgent(BaseSimulatedUserAgent):
        def generate_message(self, context: SimulatorContext) -> str:
            captured_contexts.append(context)
            return f"Message for turn {context.turn}"

    simulation_mocks["invoke"].return_value = '{"rationale": "Not yet", "result": "no"}'

    simulator = ConversationSimulator(
        test_cases=[simple_test_case],
        max_turns=2,
        user_agent_class=ContextCapturingAgent,
    )

    simulator.simulate(mock_predict_fn)

    assert len(captured_contexts) == 2
    assert captured_contexts[0].turn == 0
    assert captured_contexts[0].is_first_turn is True
    assert captured_contexts[0].goal == simple_test_case["goal"]
    assert captured_contexts[1].turn == 1
    assert captured_contexts[1].is_first_turn is False


def test_conversation_simulator_sets_span_attributes(mock_predict_fn_with_context):
    long_goal = "A" * 500
    long_persona = "B" * 500
    context = {"user_id": "U001", "session_id": "S001"}

    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.side_effect = [
            "Test message",
            '{"rationale": "Not achieved", "result": "no"}',
            "Follow up message",
            '{"rationale": "Goal achieved!", "result": "yes"}',
        ]

        simulator = ConversationSimulator(
            test_cases=[{"goal": long_goal, "persona": long_persona, "context": context}],
            max_turns=2,
        )

        all_traces = simulator.simulate(mock_predict_fn_with_context)
        first_test_case_traces = all_traces[0]

        assert len(first_test_case_traces) == 2

        for trace in first_test_case_traces:
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

        all_traces = simulator.simulate(mock_predict_fn)

        trace = all_traces[0][0]
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

        all_traces = simulator.simulate(mock_predict_fn)

        assert len(all_traces[0]) == 2

        first_trace = all_traces[0][0]
        expectation_assessments = [
            a for a in first_trace.info.assessments if a.expectation is not None
        ]

        assert len(expectation_assessments) == 2
        for assessment in expectation_assessments:
            assert assessment.name in expectations
            assert assessment.expectation.value == expectations[assessment.name]
            assert TraceMetadataKey.TRACE_SESSION in assessment.metadata

        second_trace = all_traces[0][1]
        second_trace_assessments = second_trace.info.assessments
        second_expectation_assessments = [
            a for a in second_trace_assessments if a.expectation is not None
        ]
        assert len(second_expectation_assessments) == 0


def test_invoke_llm_with_prompt_only():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "LLM response"

        agent = SimulatedUserAgent()
        result = agent.invoke_llm("Test prompt")

        assert result == "LLM response"
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Test prompt"


def test_invoke_llm_with_system_prompt():
    with patch("mlflow.genai.simulators.simulator._invoke_model_without_tracing") as mock_invoke:
        mock_invoke.return_value = "LLM response with system"

        agent = SimulatedUserAgent()
        result = agent.invoke_llm("Test prompt", system_prompt="System instructions")

        assert result == "LLM response with system"
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "System instructions"
        assert messages[1].role == "user"
        assert messages[1].content == "Test prompt"


def test_invalid_user_agent_class_raises_type_error(simple_test_case):
    class NotAUserAgent:
        pass

    with pytest.raises(TypeError, match="must be a subclass of BaseSimulatedUserAgent"):
        ConversationSimulator(
            test_cases=[simple_test_case],
            max_turns=2,
            user_agent_class=NotAUserAgent,
        )
