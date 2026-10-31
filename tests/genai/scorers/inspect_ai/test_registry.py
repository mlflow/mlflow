"""Tests for mlflow.genai.scorers.inspect_ai.registry.

Verifies callable resolution, error wrapping, and session-level task detection.
"""
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.inspect_ai.registry import get_task_callable, is_session_level_task


def test_get_task_callable_delegates_to_adapter():
    """get_task_callable delegates to adapter.find_task_callable and returns its result."""
    mock_callable = mock.MagicMock()
    with mock.patch("mlflow.genai.scorers.inspect_ai.registry.adapter.find_task_callable") as mock_find:
        mock_find.return_value = mock_callable
        result = get_task_callable("my_task")
        assert result is mock_callable
        mock_find.assert_called_once_with("my_task")


def test_get_task_callable_raises_on_not_found():
    """MlflowException from the adapter is re-raised unchanged."""
    with mock.patch(
        "mlflow.genai.scorers.inspect_ai.registry.adapter.find_task_callable",
        side_effect=MlflowException("Task not found"),
    ):
        with pytest.raises(MlflowException, match="Task not found"):
            get_task_callable("nonexistent_task")


def test_get_task_callable_wraps_unexpected_exceptions():
    """Non-MlflowException errors are wrapped with a descriptive MlflowException message."""
    with mock.patch(
        "mlflow.genai.scorers.inspect_ai.registry.adapter.find_task_callable",
        side_effect=RuntimeError("Adapter error"),
    ):
        with pytest.raises(MlflowException, match="Failed to resolve Inspect AI task"):
            get_task_callable("some_task")


def test_is_session_level_task_delegates_to_adapter():
    """is_session_level_task resolves the callable via the adapter before inspecting it."""
    with mock.patch(
        "mlflow.genai.scorers.inspect_ai.registry.adapter.find_task_callable"
    ) as mock_find:
        mock_task = mock.MagicMock()
        mock_task._metadata = {"is_conversational": True}
        mock_find.return_value = mock_task

        result = is_session_level_task("my_task")
        assert isinstance(result, bool)


def test_is_session_level_task_checks_metadata_flags():
    with mock.patch(
        "mlflow.genai.scorers.inspect_ai.registry.adapter.find_task_callable"
    ) as mock_find:
        mock_task = mock.MagicMock(spec=["is_conversational"])
        mock_task.is_conversational = True
        mock_find.return_value = mock_task

        result = is_session_level_task("conversational_task")
        assert result is True


def test_is_session_level_task_defaults_to_false_for_unknown():
    with mock.patch(
        "mlflow.genai.scorers.inspect_ai.registry.adapter.find_task_callable"
    ) as mock_find:
        mock_task = mock.MagicMock(spec=[])
        mock_find.return_value = mock_task

        result = is_session_level_task("unknown_task")
        assert result is False
