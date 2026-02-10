from unittest import mock

import pydantic
import pytest

from mlflow.entities.session import Session
from mlflow.genai.simulators.distillation import (
    _distill_goal_and_persona,
    _GoalAndPersona,
    generate_test_cases,
)


@pytest.fixture
def mock_session():
    trace = mock.MagicMock()
    return Session([trace])


def test_goal_and_persona_model_goal_required():
    with pytest.raises(pydantic.ValidationError, match="goal"):
        _GoalAndPersona.model_validate({})


@pytest.mark.parametrize(
    ("input_data", "expected_goal", "expected_persona"),
    [
        ({"goal": "Test goal"}, "Test goal", None),
        ({"goal": "Test goal", "persona": "Friendly user"}, "Test goal", "Friendly user"),
    ],
)
def test_goal_and_persona_model_validation(input_data, expected_goal, expected_persona):
    result = _GoalAndPersona.model_validate(input_data)
    assert result.goal == expected_goal
    assert result.persona == expected_persona


def test_distill_returns_none_for_empty_conversation(mock_session):
    with mock.patch(
        "mlflow.genai.simulators.distillation.resolve_conversation_from_session",
        return_value=[],
    ):
        result = _distill_goal_and_persona(mock_session, model="openai:/gpt-4o")
        assert result is None


@pytest.mark.parametrize(
    ("llm_response", "expected"),
    [
        (
            '{"goal": "Get help", "persona": "Data scientist"}',
            {"goal": "Get help", "persona": "Data scientist"},
        ),
        ('{"goal": "Get help"}', {"goal": "Get help"}),
    ],
)
def test_distill_extracts_goal_and_persona(mock_session, llm_response, expected):
    with (
        mock.patch(
            "mlflow.genai.simulators.distillation.resolve_conversation_from_session",
            return_value=[{"role": "user", "content": "Hello"}],
        ),
        mock.patch(
            "mlflow.genai.simulators.distillation.invoke_model_without_tracing",
            return_value=llm_response,
        ),
    ):
        result = _distill_goal_and_persona(mock_session, model="openai:/gpt-4o")
        assert result == expected


@pytest.mark.parametrize(
    "llm_response",
    [
        '{"goal": "", "persona": "Test"}',  # empty goal
        "invalid json",  # validation error
    ],
)
def test_distill_returns_none_for_invalid_response(mock_session, llm_response):
    with (
        mock.patch(
            "mlflow.genai.simulators.distillation.resolve_conversation_from_session",
            return_value=[{"role": "user", "content": "Hello"}],
        ),
        mock.patch(
            "mlflow.genai.simulators.distillation.invoke_model_without_tracing",
            return_value=llm_response,
        ),
    ):
        result = _distill_goal_and_persona(mock_session, model="openai:/gpt-4o")
        assert result is None


def test_generate_test_cases_processes_multiple_sessions():
    sessions = [mock.MagicMock(), mock.MagicMock()]

    with mock.patch(
        "mlflow.genai.simulators.distillation._distill_goal_and_persona",
        side_effect=[
            {"goal": "Goal 1", "persona": "Persona 1"},
            {"goal": "Goal 2"},
        ],
    ) as mock_distill:
        result = generate_test_cases(sessions, model="openai:/gpt-4o")

        assert len(result) == 2
        assert result[0] == {"goal": "Goal 1", "persona": "Persona 1"}
        assert result[1] == {"goal": "Goal 2"}
        assert mock_distill.call_count == 2


def test_generate_test_cases_filters_out_none_results():
    sessions = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]

    with mock.patch(
        "mlflow.genai.simulators.distillation._distill_goal_and_persona",
        side_effect=[{"goal": "Goal 1"}, None, {"goal": "Goal 3"}],
    ):
        result = generate_test_cases(sessions, model="openai:/gpt-4o")

        assert len(result) == 2
        assert result[0] == {"goal": "Goal 1"}
        assert result[1] == {"goal": "Goal 3"}


def test_generate_test_cases_uses_default_model_when_not_specified():
    sessions = [mock.MagicMock()]

    with (
        mock.patch(
            "mlflow.genai.simulators.distillation.get_default_simulation_model",
            return_value="openai:/gpt-5",
        ) as mock_get_model,
        mock.patch(
            "mlflow.genai.simulators.distillation._distill_goal_and_persona",
            return_value={"goal": "Test"},
        ) as mock_distill,
    ):
        generate_test_cases(sessions)

        mock_get_model.assert_called_once()
        mock_distill.assert_called_once_with(sessions[0], "openai:/gpt-5")


def test_generate_test_cases_handles_exceptions_gracefully():
    sessions = [mock.MagicMock(), mock.MagicMock()]

    with mock.patch(
        "mlflow.genai.simulators.distillation._distill_goal_and_persona",
        side_effect=[Exception("Test error"), {"goal": "Goal 2"}],
    ):
        result = generate_test_cases(sessions, model="openai:/gpt-4o")

        assert len(result) == 1
        assert result[0] == {"goal": "Goal 2"}


def test_distill_accepts_list_of_traces():
    traces = [mock.MagicMock(), mock.MagicMock()]

    with (
        mock.patch(
            "mlflow.genai.simulators.distillation.resolve_conversation_from_session",
            return_value=[{"role": "user", "content": "Hello"}],
        ) as mock_resolve,
        mock.patch(
            "mlflow.genai.simulators.distillation.invoke_model_without_tracing",
            return_value='{"goal": "Get help"}',
        ),
    ):
        result = _distill_goal_and_persona(traces, model="openai:/gpt-4o")

        assert result == {"goal": "Get help"}
        mock_resolve.assert_called_once_with(traces)


def test_generate_test_cases_accepts_list_of_trace_lists():
    trace1 = mock.MagicMock()
    trace2 = mock.MagicMock()
    sessions = [[trace1], [trace2]]

    with mock.patch(
        "mlflow.genai.simulators.distillation._distill_goal_and_persona",
        side_effect=[{"goal": "Goal 1"}, {"goal": "Goal 2"}],
    ) as mock_distill:
        result = generate_test_cases(sessions, model="openai:/gpt-4o")

        assert len(result) == 2
        assert mock_distill.call_count == 2
        mock_distill.assert_any_call([trace1], "openai:/gpt-4o")
        mock_distill.assert_any_call([trace2], "openai:/gpt-4o")
