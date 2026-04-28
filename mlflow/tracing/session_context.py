"""
Session context management for MLflow tracing.

This module provides context-aware session ID management for grouping related
traces into logical conversations or sessions. Session IDs can be set explicitly
or auto-generated, and are automatically propagated to all traces created within
the session context.
"""

from __future__ import annotations

import contextlib
import uuid
from contextvars import ContextVar
from typing import Generator

# Thread-local storage for the current chat session ID
_CURRENT_SESSION_ID: ContextVar[str | None] = ContextVar("mlflow_chat_session_id", default=None)


def get_session_id() -> str | None:
    """
    Get the current session ID from the context.

    Returns:
        The current session ID if set, otherwise None.

    Example:
        .. code-block:: python

            import mlflow

            # Within a session context
            with mlflow.set_session("my-session"):
                session_id = mlflow.get_session_id()
                print(session_id)  # "my-session"

            # Outside of session context
            session_id = mlflow.get_session_id()
            print(session_id)  # None
    """
    return _CURRENT_SESSION_ID.get()


def set_session_id(session_id: str | None) -> None:
    """
    Set the session ID for the current context.

    This function sets a session ID that will be automatically attached to all
    traces created in the current context. Use this when you want to group
    multiple traces from a multi-turn conversation or related operations.

    Args:
        session_id: The session ID to set. Pass None to clear the session ID.

    Example:
        .. code-block:: python

            import mlflow

            # Set a session ID for all subsequent traces
            mlflow.set_session_id("conversation-123")


            @mlflow.trace
            def my_function():
                # This trace will have session_id = "conversation-123"
                return "result"


            my_function()

            # Clear the session ID
            mlflow.set_session_id(None)
    """
    _CURRENT_SESSION_ID.set(session_id)


@contextlib.contextmanager
def set_session(
    session_id: str | None = None,
    auto_generate: bool = True,
) -> Generator[str, None, None]:
    """
    Context manager to set the session ID for all traces created within the context.

    This context manager provides a convenient way to group related traces together
    by setting a session ID that is automatically propagated to all traces created
    within its scope. When the context exits, the previous session ID (or None) is
    restored.

    Args:
        session_id: The session ID to use. If not provided and auto_generate is True,
            a new UUID will be generated. If not provided and auto_generate is False,
            no session ID will be set.
        auto_generate: If True and session_id is not provided, automatically generate
            a new UUID as the session ID. Default is True.

    Yields:
        The session ID being used (either provided or auto-generated).

    Example:
        .. code-block:: python

            import mlflow

            # With explicit session ID
            with mlflow.set_session("conversation-123") as session_id:

                @mlflow.trace
                def turn_1():
                    return "Hello"

                @mlflow.trace
                def turn_2():
                    return "How are you?"

                turn_1()
                turn_2()
                # Both traces will have session_id = "conversation-123"

            # With auto-generated session ID
            with mlflow.set_session() as session_id:
                print(f"Auto-generated session: {session_id}")

                @mlflow.trace
                def my_func():
                    return "result"

                my_func()
                # Trace will have the auto-generated session ID

            # Disable auto-generation
            with mlflow.set_session(auto_generate=False) as session_id:
                print(session_id)  # None
    """
    # Determine the session ID to use
    if session_id is None and auto_generate:
        session_id = uuid.uuid4().hex

    # Save the current session ID to restore later
    token = _CURRENT_SESSION_ID.set(session_id)

    try:
        yield session_id
    finally:
        # Restore the previous session ID
        _CURRENT_SESSION_ID.reset(token)


def _get_session_id_for_trace() -> str | None:
    """
    Internal function to get the session ID to be attached to a trace.

    This is used by the tracing infrastructure to automatically attach
    the session ID to traces when they are created.

    Returns:
        The current session ID if set, otherwise None.
    """
    return _CURRENT_SESSION_ID.get()
