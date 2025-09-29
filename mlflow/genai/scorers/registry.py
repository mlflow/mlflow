"""
Registered scorer functionality for MLflow GenAI.

This module provides functions to manage registered scorers that automatically
evaluate traces in MLflow experiments.
"""

import json
import warnings
from abc import ABCMeta, abstractmethod

from mlflow.exceptions import MlflowException
from mlflow.genai.scheduled_scorers import ScorerScheduleConfig
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.annotations import experimental
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.uri import get_uri_scheme


class UnsupportedScorerStoreURIException(MlflowException):
    """Exception thrown when building a scorer store with an unsupported URI"""

    def __init__(self, unsupported_uri, supported_uri_schemes):
        message = (
            f"Scorer registration functionality is unavailable; got unsupported URI"
            f" '{unsupported_uri}' for scorer data storage. Supported URI schemes are:"
            f" {supported_uri_schemes}."
        )
        super().__init__(message)
        self.supported_uri_schemes = supported_uri_schemes


class AbstractScorerStore(metaclass=ABCMeta):
    """
    Abstract class defining the interface for scorer store implementations.

    This class defines the API interface for scorer operations that can be implemented
    by different backend stores (e.g., MLflow tracking store, Databricks API).
    """

    @abstractmethod
    def register_scorer(self, experiment_id: str | None, scorer: Scorer) -> int | None:
        """
        Register a scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            scorer: The scorer object.

        Returns:
            The registered scorer version. If versioning is not supported, return None.
        """

    @abstractmethod
    def list_scorers(self, experiment_id) -> list["Scorer"]:
        """
        List all scorers for an experiment.

        Args:
            experiment_id: The experiment ID.

        Returns:
            List of mlflow.genai.scorers.Scorer objects (latest version for each scorer name).
        """

    @abstractmethod
    def get_scorer(self, experiment_id, name, version=None) -> "Scorer":
        """
        Get a specific scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            version: The scorer version. If None, returns the scorer with maximum version.

        Returns:
            A list of tuple, each tuple contains `mlflow.genai.scorers.Scorer` object.

        Raises:
            mlflow.MlflowException: If scorer is not found.
        """

    @abstractmethod
    def list_scorer_versions(self, experiment_id, name) -> list[tuple["Scorer", int]]:
        """
        List all versions of a specific scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.

        Returns:
            A list of tuple, each tuple contains `mlflow.genai.scorers.Scorer` object
            and the version number.

        Raises:
            mlflow.MlflowException: If scorer is not found.
        """

    @abstractmethod
    def delete_scorer(self, experiment_id, name, version):
        """
        Delete a scorer by name and optional version.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            version: The scorer version to delete.

        Raises:
            mlflow.MlflowException: If scorer is not found.
        """


class ScorerStoreRegistry:
    """
    Scheme-based registry for scorer store implementations.

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.scorer_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        self._registry = {}
        self.group_name = "mlflow.scorer_store"

    def register(self, scheme, store_builder):
        self._registry[scheme] = store_builder

    def register_entrypoints(self):
        """Register scorer stores provided by other packages"""
        for entrypoint in get_entry_points(self.group_name):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register scorer store for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def get_store_builder(self, store_uri):
        """Get a store from the registry based on the scheme of store_uri

        Args:
            store_uri: The store URI. If None, it will be inferred from the environment. This
                URI is used to select which scorer store implementation to instantiate
                and is passed to the constructor of the implementation.

        Returns:
            A function that returns an instance of
            ``mlflow.genai.scorers.registry.AbstractScorerStore`` that fulfills the store
            URI requirements.
        """
        scheme = store_uri if store_uri == "databricks" else get_uri_scheme(store_uri)
        try:
            store_builder = self._registry[scheme]
        except KeyError:
            raise UnsupportedScorerStoreURIException(
                unsupported_uri=store_uri, supported_uri_schemes=list(self._registry.keys())
            )
        return store_builder

    def get_store(self, tracking_uri=None):
        from mlflow.tracking._tracking_service import utils

        resolved_store_uri = utils._resolve_tracking_uri(tracking_uri)
        builder = self.get_store_builder(resolved_store_uri)
        return builder(tracking_uri=resolved_store_uri)


class MlflowTrackingStore(AbstractScorerStore):
    """
    MLflow tracking store that provides scorer functionality through the tracking store.
    This store delegates all scorer operations to the underlying tracking store.
    """

    def __init__(self, tracking_uri=None):
        self._tracking_store = _get_store(tracking_uri)

    def register_scorer(self, experiment_id: str | None, scorer: Scorer) -> int | None:
        serialized_scorer = json.dumps(scorer.model_dump())
        experiment_id = experiment_id or _get_experiment_id()
        return self._tracking_store.register_scorer(experiment_id, scorer.name, serialized_scorer)

    def list_scorers(self, experiment_id) -> list["Scorer"]:
        from mlflow.genai.scorers import Scorer

        experiment_id = experiment_id or _get_experiment_id()

        # Get ScorerVersion entities from tracking store
        scorer_versions = self._tracking_store.list_scorers(experiment_id)

        # Convert to Scorer objects
        scorers = []
        for scorer_version in scorer_versions:
            scorer = Scorer.model_validate(scorer_version.serialized_scorer)
            scorers.append(scorer)

        return scorers

    def get_scorer(self, experiment_id, name, version=None) -> "Scorer":
        from mlflow.genai.scorers import Scorer

        experiment_id = experiment_id or _get_experiment_id()

        # Get ScorerVersion entity from tracking store
        scorer_version = self._tracking_store.get_scorer(experiment_id, name, version)

        # Convert to Scorer object
        return Scorer.model_validate(scorer_version.serialized_scorer)

    def list_scorer_versions(self, experiment_id, name) -> list[tuple[Scorer, int]]:
        from mlflow.genai.scorers import Scorer

        experiment_id = experiment_id or _get_experiment_id()

        # Get ScorerVersion entities from tracking store
        scorer_versions = self._tracking_store.list_scorer_versions(experiment_id, name)

        # Convert to Scorer objects
        scorers = []
        for scorer_version in scorer_versions:
            scorer = Scorer.model_validate(scorer_version.serialized_scorer)
            version = scorer_version.scorer_version
            scorers.append((scorer, version))

        return scorers

    def delete_scorer(self, experiment_id, name, version):
        if version is None:
            raise MlflowException.invalid_parameter_value(
                "You must set `version` argument to either an integer or 'all'."
            )
        if version == "all":
            version = None

        experiment_id = experiment_id or _get_experiment_id()
        return self._tracking_store.delete_scorer(experiment_id, name, version)


class DatabricksStore(AbstractScorerStore):
    """
    Databricks store that provides scorer functionality through the Databricks API.
    This store delegates all scorer operations to the Databricks agents API.
    """

    def __init__(self, tracking_uri=None):
        pass

    @staticmethod
    def _scheduled_scorer_to_scorer(scheduled_scorer: ScorerScheduleConfig) -> Scorer:
        scorer = scheduled_scorer.scorer
        scorer._sampling_config = ScorerSamplingConfig(
            sample_rate=scheduled_scorer.sample_rate,
            filter_string=scheduled_scorer.filter_string,
        )
        return scorer

    # Private functions for internal use by Scorer methods
    @staticmethod
    def add_registered_scorer(
        *,
        name: str,
        scorer: Scorer,
        sample_rate: float,
        filter_string: str | None = None,
        experiment_id: str | None = None,
    ) -> Scorer:
        """Internal function to add a registered scorer."""
        try:
            from databricks.agents.scorers import add_scheduled_scorer
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        scheduled_scorer = add_scheduled_scorer(
            experiment_id=experiment_id,
            scheduled_scorer_name=name,
            scorer=scorer,
            sample_rate=sample_rate,
            filter_string=filter_string,
        )
        return DatabricksStore._scheduled_scorer_to_scorer(scheduled_scorer)

    @staticmethod
    def list_scheduled_scorers(experiment_id):
        try:
            from databricks.agents.scorers import list_scheduled_scorers
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        return list_scheduled_scorers(experiment_id=experiment_id)

    @staticmethod
    def get_scheduled_scorer(name, experiment_id):
        try:
            from databricks.agents.scorers import get_scheduled_scorer
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        return get_scheduled_scorer(
            scheduled_scorer_name=name,
            experiment_id=experiment_id,
        )

    @staticmethod
    def delete_scheduled_scorer(experiment_id, name):
        try:
            from databricks.agents.scorers import delete_scheduled_scorer
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        delete_scheduled_scorer(
            experiment_id=experiment_id,
            scheduled_scorer_name=name,
        )

    @staticmethod
    def update_registered_scorer(
        *,
        name: str,
        scorer: Scorer | None = None,
        sample_rate: float | None = None,
        filter_string: str | None = None,
        experiment_id: str | None = None,
    ) -> Scorer:
        """Internal function to update a registered scorer."""
        try:
            from databricks.agents.scorers import update_scheduled_scorer
        except ImportError as e:
            raise ImportError(_ERROR_MSG) from e

        scheduled_scorer = update_scheduled_scorer(
            experiment_id=experiment_id,
            scheduled_scorer_name=name,
            scorer=scorer,
            sample_rate=sample_rate,
            filter_string=filter_string,
        )
        return DatabricksStore._scheduled_scorer_to_scorer(scheduled_scorer)

    def register_scorer(self, experiment_id: str | None, scorer: Scorer) -> int | None:
        # Add the scorer to the server with sample_rate=0 (not actively sampling)
        DatabricksStore.add_registered_scorer(
            name=scorer.name,
            scorer=scorer,
            sample_rate=0.0,
            filter_string=None,
            experiment_id=experiment_id,
        )

        # Set the sampling config on the new instance
        scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.0, filter_string=None)

        return None

    def list_scorers(self, experiment_id) -> list["Scorer"]:
        # Get scheduled scorers from the server
        scheduled_scorers = DatabricksStore.list_scheduled_scorers(experiment_id)

        # Convert to Scorer instances with registration info
        scorers = []
        for scheduled_scorer in scheduled_scorers:
            scorers.append(DatabricksStore._scheduled_scorer_to_scorer(scheduled_scorer))

        return scorers

    def get_scorer(self, experiment_id, name, version=None) -> "Scorer":
        if version is not None:
            raise MlflowException.invalid_parameter_value(
                "Databricks does not support getting a certain version scorer."
            )

        # Get the scheduled scorer from the server
        scheduled_scorer = DatabricksStore.get_scheduled_scorer(name, experiment_id)

        # Extract the scorer and set registration fields
        return DatabricksStore._scheduled_scorer_to_scorer(scheduled_scorer)

    def list_scorer_versions(self, experiment_id, name) -> list[tuple["Scorer", int]]:
        raise MlflowException("Scorer DatabricksStore does not support versioning.")

    def delete_scorer(self, experiment_id, name, version):
        if version is not None:
            raise MlflowException.invalid_parameter_value(
                "Databricks does not support deleting a certain version scorer."
            )

        DatabricksStore.delete_scheduled_scorer(experiment_id, name)


# Create the global scorer store registry instance
_scorer_store_registry = ScorerStoreRegistry()


def _register_scorer_stores():
    """Register the default scorer store implementations"""
    from mlflow.store.db.db_types import DATABASE_ENGINES

    # Register for database schemes (these will use MlflowTrackingStore)
    for scheme in DATABASE_ENGINES + ["http", "https"]:
        _scorer_store_registry.register(scheme, MlflowTrackingStore)

    # Register Databricks store
    _scorer_store_registry.register("databricks", DatabricksStore)

    # Register entrypoints for custom implementations
    _scorer_store_registry.register_entrypoints()


# Register the default stores
_register_scorer_stores()


def _get_scorer_store(tracking_uri=None):
    """Get a scorer store from the registry"""
    return _scorer_store_registry.get_store(tracking_uri)


_ERROR_MSG = (
    "The `databricks-agents` package is required to register scorers. "
    "Please install it with `pip install databricks-agents`."
)


@experimental(version="3.2.0")
def list_scorers(*, experiment_id: str | None = None) -> list[Scorer]:
    """
    List all registered scorers for an experiment.

    This function retrieves all scorers that have been registered in the specified experiment.
    For each scorer name, only the latest version is returned.

    The function automatically determines the appropriate backend store (MLflow tracking store,
    Databricks, etc.) based on the current MLflow configuration and experiment location.

    Args:
        experiment_id (str, optional): The ID of the MLflow experiment containing the scorers.
            If None, uses the currently active experiment as determined by
            :func:`mlflow.get_experiment_by_name` or :func:`mlflow.set_experiment`.

    Returns:
        list[Scorer]: A list of Scorer objects, each representing the latest version of a
            registered scorer with its current configuration. The list may be empty if no
            scorers have been registered in the experiment.

    Raises:
        mlflow.MlflowException: If the experiment doesn't exist or if there are issues with
            the backend store connection.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import list_scorers

            # List all scorers in the current experiment
            scorers = list_scorers()

            # List all scorers in a specific experiment
            scorers = list_scorers(experiment_id="123")

            # Process the returned scorers
            for scorer in scorers:
                print(f"Scorer: {scorer.name}")

    Note:
        - Only the latest version of each scorer is returned.
        - This function works with both OSS MLflow tracking backend and Databricks backend.
    """
    store = _get_scorer_store()
    return store.list_scorers(experiment_id)


@experimental(version="3.2.0")
def list_scorer_versions(
    *, name: str, experiment_id: str | None = None
) -> list[tuple[Scorer, int | None]]:
    """
    List all versions of a specific scorer for an experiment.

    This function retrieves all versions of a scorer with the specified name from the given
    experiment.

    The function returns a list of tuples, where each tuple contains a Scorer instance and
    its corresponding version number.

    Args:
        name (str): The name of the scorer to list versions for. This must match exactly
            with the name used during scorer registration.
        experiment_id (str, optional): The ID of the MLflow experiment containing the scorer.
            If None, uses the currently active experiment as determined by
            :func:`mlflow.get_experiment_by_name` or :func:`mlflow.set_experiment`.

    Returns:
        list[tuple[Scorer, int | None]]: A list of tuples, where each tuple contains:
            - A Scorer object representing the scorer at that specific version
            - An integer representing the version number (1, 2, 3, etc.), for Databricks backend,
              the version number is `None`.
            The list may be empty if no versions of the scorer exist.

    Raises:
        mlflow.MlflowException: If the scorer with the specified name is not found in
            the experiment, if the experiment doesn't exist, or if there are issues with the backend
            store.
    """
    store = _get_scorer_store()
    return store.list_scorer_versions(experiment_id, name)


@experimental(version="3.2.0")
def get_scorer(
    *, name: str, experiment_id: str | None = None, version: int | None = None
) -> Scorer:
    """
    Retrieve a specific registered scorer by name and optional version.

    This function retrieves a single Scorer instance from the specified experiment. If no
    version is specified, it returns the latest (highest version number) scorer with the
    given name.

    Args:
        name (str): The name of the registered scorer to retrieve. This must match exactly
            with the name used during scorer registration.
        experiment_id (str, optional): The ID of the MLflow experiment containing the scorer.
            If None, uses the currently active experiment as determined by
            :func:`mlflow.get_experiment_by_name` or :func:`mlflow.set_experiment`.
        version (int, optional): The specific version of the scorer to retrieve. If None,
            returns the scorer with the highest version number (latest version).

    Returns:
        Scorer: A Scorer object representing the requested scorer.

    Raises:
        mlflow.MlflowException: If the scorer with the specified name is not found in the
            experiment, if the specified version doesn't exist, if the experiment doesn't exist,
            or if there are issues with the backend store connection.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import get_scorer

            # Get the latest version of a scorer
            latest_scorer = get_scorer(name="accuracy_scorer")

            # Get a specific version of a scorer
            v2_scorer = get_scorer(name="safety_scorer", version=2)

            # Get a scorer from a specific experiment
            scorer = get_scorer(name="relevance_scorer", experiment_id="123")

    Note:
        - When no version is specified, the function automatically returns the latest version
        - This function works with both OSS MLflow tracking backend and Databricks backend.
        - For Databricks backend, versioning is not supported, so the version parameter
          should be None.
    """

    store = _get_scorer_store()
    return store.get_scorer(experiment_id, name, version)


@experimental(version="3.2.0")
def delete_scorer(
    *,
    name: str,
    experiment_id: str | None = None,
    version: int | str | None = None,
) -> None:
    """
    Delete a registered scorer from the MLflow experiment.

    This function permanently removes scorer registrations.
    The behavior of this function varies depending on the backend store and version parameter:

    **OSS MLflow Tracking Backend:**
        - Supports versioning with granular deletion options
        - Can delete specific versions or all versions of a scorer by setting `version`
          parameter to "all"

    **Databricks Backend:**
        - Does not support versioning
        - Deletes the entire scorer regardless of version parameter
        - `version` parameter must be None

    Args:
        name (str): The name of the scorer to delete. This must match exactly with the
            name used during scorer registration.
        experiment_id (str, optional): The ID of the MLflow experiment containing the scorer.
            If None, uses the currently active experiment as determined by
            :func:`mlflow.get_experiment_by_name` or :func:`mlflow.set_experiment`.
        version (int | str | None, optional): The version(s) to delete:
            For OSS MLflow tracking backend: if `None`, deletes the latest version only, if version
            is an integer, deletes the specific version, if version is the string 'all', deletes
            all versions of the scorer
            For Databricks backend, the version must be set to `None` (versioning not supported)

    Raises:
        mlflow.MlflowException: If the scorer with the specified name is not found in
            the experiment, if the specified version doesn't exist, or if versioning
            is not supported for the current backend.

    Example:
        .. code-block:: python

            from mlflow.genai.scorers import delete_scorer

            # Delete the latest version of a scorer from current experiment
            delete_scorer(name="accuracy_scorer")

            # Delete a specific version of a scorer
            delete_scorer(name="safety_scorer", version=2)

            # Delete all versions of a scorer
            delete_scorer(name="relevance_scorer", version="all")

            # Delete a scorer from a specific experiment
            delete_scorer(name="harmfulness_scorer", experiment_id="123", version=1)
    """

    store = _get_scorer_store()
    return store.delete_scorer(experiment_id, name, version)
