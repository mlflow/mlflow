"""
Introduces main Context class and the framework to specify different specialized
contexts.
"""

import functools
from abc import ABC, abstractmethod
from typing import Callable, ParamSpec, TypeVar

import mlflow
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_USER

P = ParamSpec("P")
R = TypeVar("R")


class Context(ABC):
    """
    Abstract class for execution context.
    Context is stateless and should NOT be used to store information related to specific eval run.
    """

    @abstractmethod
    def get_mlflow_experiment_id(self) -> str | None:
        """
        Get the current MLflow experiment ID, or None if not running within an MLflow experiment.
        """

    @abstractmethod
    def get_mlflow_run_id(self) -> str | None:
        """
        Gets the MLflow RunId, or None if not running within an MLflow run.
        """

    @abstractmethod
    def get_user_name(self) -> str:
        """
        Get the current user's name.
        """


class NoneContext(Context):
    """
    A context that does nothing.
    """

    def get_mlflow_experiment_id(self) -> str | None:
        raise NotImplementedError("Context is not set")

    def get_mlflow_run_id(self) -> str | None:
        raise NotImplementedError("Context is not set")

    def get_user_name(self) -> str:
        raise NotImplementedError("Context is not set")


class RealContext(Context):
    """
    Context for eval execution.

    NOTE: This class is not covered by unit tests and is meant to be tested through
    smoke tests that run this code on an actual Databricks cluster.
    """

    def __init__(self):
        self._run_id = None
        self._context_tags = context_registry.resolve_tags()

    def get_mlflow_experiment_id(self) -> str | None:
        # Note `_get_experiment_id` is thread-safe
        return mlflow.tracking.fluent._get_experiment_id()

    def get_mlflow_run_id(self) -> str | None:
        """
        Gets the MLflow run_id the evaluation harness is running under.

        Warning: This run_id may not be active. This happens when `get_mlflow_run_id` is called from
        a different thread than the one that started the MLflow run.
        """
        # First check if a run ID is specified explicitly by the parent thread
        if self._run_id:
            return self._run_id

        # Otherwise fall back to the active run in the current thread
        if run := mlflow.active_run():
            return run.info.run_id

        return None

    def set_mlflow_run_id(self, run_id: str) -> None:
        """
        Set the MLflow run ID explicitly.

        This method should be called when running code in a different thread than the one that
        started the MLflow run. It sets the run ID in a thread-local variable so that it can be
        accessed from the thread.
        """
        self._run_id = run_id

    def get_user_name(self) -> str:
        return self._context_tags.get(MLFLOW_USER, "unknown")


# Context is a singleton.
_context_singleton = NoneContext()


def context_is_active() -> bool:
    """
    Check if a context is active.
    """
    return not isinstance(get_context(), NoneContext)


def get_context() -> Context:
    """
    Get the context.
    """
    return _context_singleton


def eval_context(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator for wrapping all eval APIs with setup and closure logic.

    Sets up a context singleton with RealContext if there isn't one already.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Set up the context singleton if it doesn't exist
        if not context_is_active():
            global _context_singleton
            _context_singleton = RealContext()

        return func(*args, **kwargs)

    return wrapper


def _set_context(context: Context) -> None:
    """SHOULD ONLY BE USED FOR TESTING."""
    global _context_singleton
    _context_singleton = context
