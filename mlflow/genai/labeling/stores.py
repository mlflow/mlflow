"""
Labeling store functionality for MLflow GenAI.

This module provides store implementations to manage labeling sessions and schemas
"""

import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from mlflow.entities import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.label_schemas.label_schemas import LabelSchema
from mlflow.genai.labeling.databricks_utils import get_databricks_review_app
from mlflow.genai.labeling.labeling import LabelingSession
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking._tracking_service import utils as tracking_utils
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.uri import get_uri_scheme

if TYPE_CHECKING:
    from databricks.agents.review_app.labeling import LabelingSession as _DatabricksLabelingSession


class UnsupportedLabelingStoreURIException(MlflowException):
    """Exception thrown when building a labeling store with an unsupported URI"""

    def __init__(self, unsupported_uri: str, supported_uri_schemes: list[str]) -> None:
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

    def __init__(self, tracking_uri: str | None = None) -> None:
        """
        Initialize the labeling store.

        Args:
            tracking_uri: The tracking URI for the store.
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

    @abstractmethod
    def add_dataset_to_session(
        self,
        labeling_session: LabelingSession,
        dataset_name: str,
        record_ids: list[str] | None = None,
    ) -> LabelingSession:
        """
        Add a dataset to a labeling session.

        Args:
            labeling_session: The labeling session to add the dataset to.
            dataset_name: The name of the dataset.
            record_ids: Optional. The individual record ids to be added to the session.

        Returns:
            LabelingSession: The updated labeling session.
        """

    @abstractmethod
    def add_traces_to_session(
        self,
        labeling_session: LabelingSession,
        traces: list[Trace],
    ) -> LabelingSession:
        """
        Add traces to a labeling session.

        Args:
            labeling_session: The labeling session to add traces to.
            traces: List of Trace objects to add.

        Returns:
            LabelingSession: The updated labeling session.
        """

    @abstractmethod
    def sync_session_expectations(self, labeling_session: LabelingSession, dataset: str) -> None:
        """
        Sync traces and expectations from a labeling session to a dataset.

        Args:
            labeling_session: The labeling session to sync.
            dataset: The name of the dataset to sync traces and expectations to.
        """

    @abstractmethod
    def set_session_assigned_users(
        self, labeling_session: LabelingSession, assigned_users: list[str]
    ) -> LabelingSession:
        """
        Set the assigned users for a labeling session.

        Args:
            labeling_session: The labeling session to update.
            assigned_users: The list of users to assign to the session.

        Returns:
            LabelingSession: The updated labeling session.
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

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., AbstractLabelingStore]] = {}
        self.group_name = "mlflow.labeling_store"

    def register(self, scheme: str, store_builder: Callable[..., AbstractLabelingStore]) -> None:
        self._registry[scheme] = store_builder

    def register_entrypoints(self) -> None:
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

    def get_store_builder(self, store_uri: str) -> Callable[..., AbstractLabelingStore]:
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

    def get_store(self, tracking_uri: str | None = None) -> AbstractLabelingStore:
        resolved_store_uri = tracking_utils._resolve_tracking_uri(tracking_uri)
        builder = self.get_store_builder(resolved_store_uri)
        return builder(tracking_uri=resolved_store_uri)


class DatabricksLabelingStore(AbstractLabelingStore):
    """
    Databricks store that provides labeling functionality through the Databricks API.
    This store delegates all labeling operations to the Databricks agents API.
    """

    def _get_backend_session(
        self, labeling_session: LabelingSession
    ) -> "_DatabricksLabelingSession":
        """
        Get the backend session for a labeling session.

        Note: We have to list all sessions and match by ID because the Databricks
        agents API doesn't provide a direct get/fetch API for individual labeling sessions.
        """
        app = get_databricks_review_app(labeling_session.experiment_id)
        backend_sessions = app.get_labeling_sessions()
        backend_session = next(
            (
                session
                for session in backend_sessions
                if session.labeling_session_id == labeling_session.labeling_session_id
            ),
            None,
        )
        if backend_session is None:
            raise MlflowException(
                f"Labeling session {labeling_session.labeling_session_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return backend_session

    def _databricks_session_to_labeling_session(
        self, databricks_session: "_DatabricksLabelingSession"
    ) -> LabelingSession:
        """Create a LabelingSession from a Databricks backend session object."""
        return LabelingSession(
            name=databricks_session.name,
            assigned_users=databricks_session.assigned_users,
            agent=databricks_session.agent,
            label_schemas=databricks_session.label_schemas,
            labeling_session_id=databricks_session.labeling_session_id,
            mlflow_run_id=databricks_session.mlflow_run_id,
            review_app_id=databricks_session.review_app_id,
            experiment_id=databricks_session.experiment_id,
            url=databricks_session.url,
            enable_multi_turn_chat=databricks_session.enable_multi_turn_chat,
            custom_inputs=databricks_session.custom_inputs,
        )

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
        app = get_databricks_review_app(experiment_id)
        sessions = app.get_labeling_sessions()
        return [self._databricks_session_to_labeling_session(session) for session in sessions]

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
        app = get_databricks_review_app(experiment_id)
        backend_session = app.create_labeling_session(
            name=name,
            assigned_users=assigned_users or [],
            agent=agent,
            label_schemas=label_schemas or [],
            enable_multi_turn_chat=enable_multi_turn_chat,
            custom_inputs=custom_inputs,
        )
        return self._databricks_session_to_labeling_session(backend_session)

    def delete_labeling_session(self, labeling_session: LabelingSession) -> None:
        """Delete a labeling session."""
        backend_session = self._get_backend_session(labeling_session)
        app = get_databricks_review_app(labeling_session.experiment_id)
        app.delete_labeling_session(backend_session)

    def get_label_schema(self, name: str) -> LabelSchema:
        """Get a label schema by name."""
        app = get_databricks_review_app()
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
        app = get_databricks_review_app()
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
        app = get_databricks_review_app()
        app.delete_label_schema(name)

    def add_dataset_to_session(
        self,
        labeling_session: LabelingSession,
        dataset_name: str,
        record_ids: list[str] | None = None,
    ) -> LabelingSession:
        """Add a dataset to a labeling session."""
        backend_session = self._get_backend_session(labeling_session)
        updated_session = backend_session.add_dataset(dataset_name, record_ids)
        return self._databricks_session_to_labeling_session(updated_session)

    def add_traces_to_session(
        self,
        labeling_session: LabelingSession,
        traces: list[Trace],
    ) -> LabelingSession:
        """Add traces to a labeling session."""
        backend_session = self._get_backend_session(labeling_session)
        updated_session = backend_session.add_traces(traces)
        return self._databricks_session_to_labeling_session(updated_session)

    def sync_session_expectations(self, labeling_session: LabelingSession, dataset: str) -> None:
        """Sync traces and expectations from a labeling session to a dataset."""
        backend_session = self._get_backend_session(labeling_session)
        backend_session.sync_expectations(dataset)

    def set_session_assigned_users(
        self, labeling_session: LabelingSession, assigned_users: list[str]
    ) -> LabelingSession:
        """Set the assigned users for a labeling session."""
        backend_session = self._get_backend_session(labeling_session)
        updated_session = backend_session.set_assigned_users(assigned_users)
        return self._databricks_session_to_labeling_session(updated_session)


# Create the global labeling store registry instance
_labeling_store_registry = LabelingStoreRegistry()


def _register_labeling_stores() -> None:
    """Register the default labeling store implementations"""
    # Register Databricks store
    _labeling_store_registry.register("databricks", DatabricksLabelingStore)

    # Register entrypoints for custom implementations
    _labeling_store_registry.register_entrypoints()


# Register the default stores
_register_labeling_stores()


def _get_labeling_store(tracking_uri: str | None = None) -> AbstractLabelingStore:
    """Get a labeling store from the registry"""
    return _labeling_store_registry.get_store(tracking_uri)
