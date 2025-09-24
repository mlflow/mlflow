"""
Labeling store functionality for MLflow GenAI.

This module provides store implementations to manage labeling sessions and schemas
in MLflow experiments, similar to how scorer stores work.
"""

import warnings
from abc import ABCMeta, abstractmethod
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas.label_schemas import LabelSchema
from mlflow.genai.labeling.labeling import LabelingSession
from mlflow.utils.annotations import experimental
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.uri import get_uri_scheme


class UnsupportedLabelingStoreURIException(MlflowException):
    """Exception thrown when building a labeling store with an unsupported URI"""

    def __init__(self, unsupported_uri, supported_uri_schemes):
        message = (
            f"Labeling functionality is unavailable; got unsupported URI"
            f" '{unsupported_uri}' for labeling data storage. Supported URI schemes are:"
            f" {supported_uri_schemes}."
        )
        super().__init__(message)
        self.supported_uri_schemes = supported_uri_schemes


class AbstractLabelingStore(metaclass=ABCMeta):
    """
    Abstract class defining the interface for labeling store implementations.

    This class defines the API interface for labeling operations that can be implemented
    by different backend stores (e.g., MLflow tracking store, Databricks API).
    """

    @abstractmethod
    def get_labeling_session(self, run_id: str) -> LabelingSession:
        """
        Get a labeling session by MLflow run ID.

        Args:
            run_id: The MLflow run ID of the labeling session.

        Returns:
            LabelingSession: The labeling session.

        Raises:
            mlflow.MlflowException: If labeling session is not found.
        """

    @abstractmethod
    def get_labeling_sessions(self, experiment_id: str | None = None) -> list[LabelingSession]:
        """
        Get all labeling sessions for an experiment.

        Args:
            experiment_id: The experiment ID. If None, uses the currently active experiment.

        Returns:
            list[LabelingSession]: List of labeling sessions.
        """

    @abstractmethod
    def create_labeling_session(
        self,
        name: str,
        *,
        assigned_users: list[str] | None = None,
        agent: str | None = None,
        label_schemas: list[str] | None = None,
        enable_multi_turn_chat: bool = False,
        custom_inputs: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> LabelingSession:
        """
        Create a new labeling session.

        Args:
            name: The name of the labeling session.
            assigned_users: The users that will be assigned to label items in the session.
            agent: The agent to be used to generate responses for the items in the session.
            label_schemas: The label schemas to be used in the session.
            enable_multi_turn_chat: Whether to enable multi-turn chat labeling for the session.
            custom_inputs: Optional. Custom inputs to be used in the session.
            experiment_id: The experiment ID. If None, uses the currently active experiment.

        Returns:
            LabelingSession: The created labeling session.
        """

    @abstractmethod
    def delete_labeling_session(self, labeling_session: LabelingSession) -> None:
        """
        Delete a labeling session.

        Args:
            labeling_session: The labeling session to delete.
        """

    @abstractmethod
    def get_label_schema(self, name: str) -> LabelSchema:
        """
        Get a label schema by name.

        Args:
            name: The name of the label schema.

        Returns:
            LabelSchema: The label schema.

        Raises:
            mlflow.MlflowException: If label schema is not found.
        """

    @abstractmethod
    def create_label_schema(
        self,
        name: str,
        *,
        type: str,
        title: str,
        input: Any,
        instruction: str | None = None,
        enable_comment: bool = False,
        overwrite: bool = False,
    ) -> LabelSchema:
        """
        Create a new label schema.

        Args:
            name: The name of the label schema. Must be unique across the review app.
            type: The type of the label schema. Either "feedback" or "expectation".
            title: The title of the label schema shown to stakeholders.
            input: The input type of the label schema.
            instruction: Optional. The instruction shown to stakeholders.
            enable_comment: Optional. Whether to enable comments for the label schema.
            overwrite: Optional. Whether to overwrite the existing label schema with the same name.

        Returns:
            LabelSchema: The created label schema.
        """

    @abstractmethod
    def delete_label_schema(self, name: str) -> None:
        """
        Delete a label schema.

        Args:
            name: The name of the label schema to delete.
        """


class LabelingStoreRegistry:
    """
    Scheme-based registry for labeling store implementations.

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.labeling_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        self._registry = {}
        self.group_name = "mlflow.labeling_store"

    def register(self, scheme, store_builder):
        self._registry[scheme] = store_builder

    def register_entrypoints(self):
        """Register labeling stores provided by other packages"""
        for entrypoint in get_entry_points(self.group_name):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register labeling store for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def get_store_builder(self, store_uri):
        """Get a store from the registry based on the scheme of store_uri

        Args:
            store_uri: The store URI. If None, it will be inferred from the environment. This
                URI is used to select which labeling store implementation to instantiate
                and is passed to the constructor of the implementation.

        Returns:
            A function that returns an instance of
            ``mlflow.genai.labeling.stores.AbstractLabelingStore`` that fulfills the store
            URI requirements.
        """
        scheme = store_uri if store_uri == "databricks" else get_uri_scheme(store_uri)
        try:
            store_builder = self._registry[scheme]
        except KeyError:
            raise UnsupportedLabelingStoreURIException(
                unsupported_uri=store_uri, supported_uri_schemes=list(self._registry.keys())
            )
        return store_builder

    def get_store(self, tracking_uri=None):
        from mlflow.tracking._tracking_service import utils

        resolved_store_uri = utils._resolve_tracking_uri(tracking_uri)
        builder = self.get_store_builder(resolved_store_uri)
        return builder(tracking_uri=resolved_store_uri)


class DatabricksLabelingStore(AbstractLabelingStore):
    """
    Databricks store that provides labeling functionality through the Databricks API.
    This store delegates all labeling operations to the Databricks agents API.
    """

    def __init__(self, tracking_uri=None):
        pass

    def get_labeling_session(self, run_id: str) -> LabelingSession:
        """Get a labeling session by MLflow run ID."""
        labeling_sessions = self.get_labeling_sessions()
        labeling_session = next(
            (
                labeling_session
                for labeling_session in labeling_sessions
                if labeling_session.mlflow_run_id == run_id
            ),
            None,
        )
        if labeling_session is None:
            raise MlflowException(f"Labeling session with run_id `{run_id}` not found")
        return labeling_session

    def get_labeling_sessions(self, experiment_id: str | None = None) -> list[LabelingSession]:
        """Get all labeling sessions for an experiment."""
        try:
            from databricks.agents.review_app import get_review_app as _get_review_app
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        sessions = _get_review_app(experiment_id).get_labeling_sessions()
        return [LabelingSession(session) for session in sessions]

    def create_labeling_session(
        self,
        name: str,
        *,
        assigned_users: list[str] | None = None,
        agent: str | None = None,
        label_schemas: list[str] | None = None,
        enable_multi_turn_chat: bool = False,
        custom_inputs: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> LabelingSession:
        """Create a new labeling session."""
        try:
            from databricks.agents.review_app import get_review_app as _get_review_app
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        return LabelingSession(
            _get_review_app(experiment_id).create_labeling_session(
                name=name,
                assigned_users=assigned_users or [],
                agent=agent,
                label_schemas=label_schemas or [],
                enable_multi_turn_chat=enable_multi_turn_chat,
                custom_inputs=custom_inputs,
            )
        )

    def delete_labeling_session(self, labeling_session: LabelingSession) -> None:
        """Delete a labeling session."""
        try:
            from databricks.agents.review_app import get_review_app as _get_review_app
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        _get_review_app().delete_labeling_session(labeling_session._session)

    def get_label_schema(self, name: str) -> LabelSchema:
        """Get a label schema by name."""
        try:
            from databricks.agents import review_app
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        app = review_app.get_review_app()
        label_schema = next(
            (label_schema for label_schema in app.label_schemas if label_schema.name == name),
            None,
        )
        if label_schema is None:
            raise MlflowException(f"Label schema with name `{name}` not found")
        return LabelSchema._from_databricks_label_schema(label_schema)

    def create_label_schema(
        self,
        name: str,
        *,
        type: str,
        title: str,
        input: Any,
        instruction: str | None = None,
        enable_comment: bool = False,
        overwrite: bool = False,
    ) -> LabelSchema:
        """Create a new label schema."""
        try:
            from databricks.agents import review_app
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        app = review_app.get_review_app()
        return app.create_label_schema(
            name=name,
            type=type,
            title=title,
            input=input._to_databricks_input(),
            instruction=instruction,
            enable_comment=enable_comment,
            overwrite=overwrite,
        )

    def delete_label_schema(self, name: str) -> None:
        """Delete a label schema."""
        try:
            from databricks.agents import review_app
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        app = review_app.get_review_app()
        app.delete_label_schema(name)


# Create the global labeling store registry instance
_labeling_store_registry = LabelingStoreRegistry()


def _register_labeling_stores():
    """Register the default labeling store implementations"""
    # Register Databricks store
    _labeling_store_registry.register("databricks", DatabricksLabelingStore)

    # Register entrypoints for custom implementations
    _labeling_store_registry.register_entrypoints()


# Register the default stores
_register_labeling_stores()


def _get_labeling_store(tracking_uri=None):
    """Get a labeling store from the registry"""
    return _labeling_store_registry.get_store(tracking_uri)


_ERROR_MSG = (
    "The `databricks-agents` package is required to use labeling functionality. "
    "Please install it with `pip install databricks-agents`."
)


@experimental(version="3.3.0")
def get_labeling_session(run_id: str) -> LabelingSession:
    """
    Get a labeling session by MLflow run ID.

    This function retrieves a specific labeling session identified by its MLflow run ID.
    The function automatically determines the appropriate backend store (Databricks)
    based on the current MLflow configuration.

    Args:
        run_id (str): The MLflow run ID of the labeling session to retrieve.

    Returns:
        LabelingSession: The labeling session object.

    Raises:
        mlflow.MlflowException: If the labeling session with the specified run ID is not found.

    Example:
        .. code-block:: python

            from mlflow.genai.labeling.stores import get_labeling_session

            # Get a specific labeling session
            session = get_labeling_session("your-mlflow-run-id")
            print(f"Session name: {session.name}")

    Note:
        - This function currently only works with Databricks backend.
        - MLflow tracking store support is planned for future releases.
    """
    store = _get_labeling_store()
    return store.get_labeling_session(run_id)


@experimental(version="3.3.0")
def get_labeling_sessions(experiment_id: str | None = None) -> list[LabelingSession]:
    """
    Get all labeling sessions for an experiment.

    This function retrieves all labeling sessions that have been created in the specified
    experiment. The function automatically determines the appropriate backend store
    (Databricks) based on the current MLflow configuration.

    Args:
        experiment_id (str, optional): The ID of the MLflow experiment containing the sessions.
            If None, uses the currently active experiment as determined by
            :func:`mlflow.get_experiment_by_name` or :func:`mlflow.set_experiment`.

    Returns:
        list[LabelingSession]: A list of LabelingSession objects. The list may be empty
            if no labeling sessions have been created in the experiment.

    Raises:
        mlflow.MlflowException: If the experiment doesn't exist or if there are issues with
            the backend store connection.

    Example:
        .. code-block:: python

            from mlflow.genai.labeling.stores import get_labeling_sessions

            # Get all sessions in the current experiment
            sessions = get_labeling_sessions()

            # Get all sessions in a specific experiment
            sessions = get_labeling_sessions(experiment_id="123")

            # Process the returned sessions
            for session in sessions:
                print(f"Session: {session.name}")

    Note:
        - This function currently only works with Databricks backend.
        - MLflow tracking store support is planned for future releases.
    """
    store = _get_labeling_store()
    return store.get_labeling_sessions(experiment_id)


@experimental(version="3.3.0")
def create_labeling_session(
    name: str,
    *,
    assigned_users: list[str] | None = None,
    agent: str | None = None,
    label_schemas: list[str] | None = None,
    enable_multi_turn_chat: bool = False,
    custom_inputs: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> LabelingSession:
    """
    Create a new labeling session.

    This function creates a new labeling session in the specified experiment for collecting
    human feedback on model outputs. The session can be configured with specific users,
    agents, and label schemas.

    Args:
        name (str): The name of the labeling session.
        assigned_users (list[str], optional): The users that will be assigned to label
            items in the session. If None, no users are pre-assigned.
        agent (str, optional): The agent to be used to generate responses for the items
            in the session. If None, no agent is associated.
        label_schemas (list[str], optional): The label schemas to be used in the session.
            These define what type of feedback will be collected. If None, no schemas
            are pre-configured.
        enable_multi_turn_chat (bool): Whether to enable multi-turn chat labeling for
            the session. Defaults to False.
        custom_inputs (dict[str, Any], optional): Optional custom inputs to be used
            in the session. If None, no custom inputs are configured.
        experiment_id (str, optional): The ID of the MLflow experiment to create the
            session in. If None, uses the currently active experiment.

    Returns:
        LabelingSession: The created labeling session object.

    Raises:
        mlflow.MlflowException: If there are issues creating the session or if the
            backend store is not supported.

    Example:
        .. code-block:: python

            from mlflow.genai.labeling.stores import create_labeling_session

            # Create a basic labeling session
            session = create_labeling_session("my-feedback-session")

            # Create a session with specific configuration
            session = create_labeling_session(
                name="quality-review",
                assigned_users=["user1@company.com", "user2@company.com"],
                agent="my-chatbot-agent",
                label_schemas=["relevance", "accuracy"],
                enable_multi_turn_chat=True,
                custom_inputs={"context": "product-reviews"},
            )

    Note:
        - This function currently only works with Databricks backend.
        - MLflow tracking store support is planned for future releases.
    """
    store = _get_labeling_store()
    return store.create_labeling_session(
        name=name,
        assigned_users=assigned_users,
        agent=agent,
        label_schemas=label_schemas,
        enable_multi_turn_chat=enable_multi_turn_chat,
        custom_inputs=custom_inputs,
        experiment_id=experiment_id,
    )


@experimental(version="3.3.0")
def delete_labeling_session(labeling_session: LabelingSession) -> None:
    """
    Delete a labeling session.

    This function permanently removes a labeling session and all associated data.
    The operation cannot be undone.

    Args:
        labeling_session (LabelingSession): The labeling session to delete.

    Raises:
        mlflow.MlflowException: If the session cannot be found or if there are issues
            with the backend store connection.

    Example:
        .. code-block:: python

            from mlflow.genai.labeling.stores import get_labeling_session, delete_labeling_session

            # Get a session and delete it
            session = get_labeling_session("your-mlflow-run-id")
            delete_labeling_session(session)

    Warning:
        This operation is irreversible. All feedback and data associated with the
        labeling session will be permanently lost.

    Note:
        - This function currently only works with Databricks backend.
        - MLflow tracking store support is planned for future releases.
    """
    store = _get_labeling_store()
    return store.delete_labeling_session(labeling_session)
