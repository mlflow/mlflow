"""
Registered scorer functionality for MLflow GenAI.

This module provides functions to manage registered scorers that automatically
evaluate traces in MLflow experiments.
"""

import json
import warnings
from abc import ABCMeta, abstractmethod
from base64 import urlsafe_b64encode
from functools import partial
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import (
    SCORER_BACKEND_DATABRICKS,
    SCORER_BACKEND_TRACKING,
    Scorer,
    ScorerSamplingConfig,
)
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.rest_utils import http_request, verify_rest_response
from mlflow.utils.uri import get_uri_scheme

if TYPE_CHECKING:
    from mlflow.genai.scorers.online.entities import OnlineScoringConfig


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
            experiment_id: The ID of the Experiment containing the scorer.
            scorer: The scorer object.

        Returns:
            The registered scorer version. If versioning is not supported, return None.
        """

    @abstractmethod
    def list_scorers(self, experiment_id) -> list["Scorer"]:
        """
        List all scorers for an experiment.

        Args:
            experiment_id: The ID of the Experiment containing the scorer.

        Returns:
            List of mlflow.genai.scorers.Scorer objects (latest version for each scorer name).
        """

    @abstractmethod
    def get_scorer(self, experiment_id, name, version=None) -> "Scorer":
        """
        Get a specific scorer for an experiment.

        Args:
            experiment_id: The ID of the Experiment containing the scorer.
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
            experiment_id: The ID of the Experiment containing the scorer.
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
            experiment_id: The ID of the Experiment containing the scorer.
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
        version = self._tracking_store.register_scorer(
            experiment_id, scorer.name, serialized_scorer
        )
        self._hydrate_scorer(scorer, experiment_id, online_config=None)
        return version

    def _hydrate_scorer(
        self,
        scorer: Scorer,
        experiment_id: str,
        online_config: Optional["OnlineScoringConfig"] = None,
    ) -> None:
        """
        Hydrate a scorer with runtime state from the tracking store.

        Args:
            scorer: The scorer to hydrate.
            experiment_id: The experiment ID the scorer belongs to.
            online_config: Optional OnlineScoringConfig from the tracking store.
        """
        scorer._registered_backend = SCORER_BACKEND_TRACKING
        scorer._experiment_id = experiment_id
        if online_config is not None:
            scorer._sampling_config = ScorerSamplingConfig(
                sample_rate=online_config.sample_rate,
                filter_string=online_config.filter_string,
            )

    def list_scorers(self, experiment_id) -> list["Scorer"]:
        from mlflow.genai.scorers import Scorer

        experiment_id = experiment_id or _get_experiment_id()
        scorer_versions = self._tracking_store.list_scorers(experiment_id)
        scorer_ids = [sv.scorer_id for sv in scorer_versions]
        online_configs_list = (
            self._tracking_store.get_online_scoring_configs(scorer_ids) if scorer_ids else []
        )
        # Each scorer has at most one online configuration, guaranteed by the server
        online_configs = {c.scorer_id: c for c in online_configs_list}
        scorers = []
        for scorer_version in scorer_versions:
            scorer = Scorer.model_validate(scorer_version.serialized_scorer)
            online_config = online_configs.get(scorer_version.scorer_id)
            self._hydrate_scorer(scorer, experiment_id, online_config)
            scorers.append(scorer)
        return scorers

    def get_scorer(self, experiment_id, name, version=None) -> "Scorer":
        from mlflow.genai.scorers import Scorer

        experiment_id = experiment_id or _get_experiment_id()
        scorer_version = self._tracking_store.get_scorer(experiment_id, name, version)
        online_configs_list = self._tracking_store.get_online_scoring_configs([
            scorer_version.scorer_id
        ])
        # Each scorer has at most one online configuration, guaranteed by the server
        online_config = online_configs_list[0] if online_configs_list else None
        scorer = Scorer.model_validate(scorer_version.serialized_scorer)
        self._hydrate_scorer(scorer, experiment_id, online_config)
        return scorer

    def list_scorer_versions(self, experiment_id, name) -> list[tuple[Scorer, int]]:
        from mlflow.genai.scorers import Scorer

        experiment_id = experiment_id or _get_experiment_id()
        scorer_versions = self._tracking_store.list_scorer_versions(experiment_id, name)
        scorer_ids = list({sv.scorer_id for sv in scorer_versions})
        online_configs_list = (
            self._tracking_store.get_online_scoring_configs(scorer_ids) if scorer_ids else []
        )
        # Each scorer has at most one online configuration, guaranteed by the server
        online_configs = {c.scorer_id: c for c in online_configs_list}
        scorers = []
        for scorer_version in scorer_versions:
            scorer = Scorer.model_validate(scorer_version.serialized_scorer)
            online_config = online_configs.get(scorer_version.scorer_id)
            self._hydrate_scorer(scorer, experiment_id, online_config)
            scorers.append((scorer, scorer_version.scorer_version))
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

    def upsert_online_scoring_config(
        self,
        *,
        scorer: Scorer,
        experiment_id: str,
        sample_rate: float,
        filter_string: str | None = None,
    ) -> Scorer:
        """
        Create or update the online scoring configuration for a registered scorer.

        Args:
            scorer: The scorer instance to update.
            experiment_id: The ID of the MLflow experiment containing the scorer.
            sample_rate: The sampling rate (0.0 to 1.0).
            filter_string: Optional filter string.

        Returns:
            A copy of the scorer with updated sampling configuration.

        Raises:
            MlflowException: If the scorer is not registered.
        """
        if scorer._registered_backend is None:
            raise MlflowException.invalid_parameter_value(
                "Cannot start/update a scorer that is not registered. "
                "Please call register() first before calling start()/update(), "
                "or use get_scorer() to load a registered scorer."
            )

        self._tracking_store.upsert_online_scoring_config(
            experiment_id=experiment_id,
            scorer_name=scorer.name,
            sample_rate=sample_rate,
            filter_string=filter_string,
        )

        return self.get_scorer(experiment_id, scorer.name)


class DatabricksStore(AbstractScorerStore):
    """
    Databricks store that provides scorer functionality through the Databricks API.
    This store delegates scorer registry operations to the managed-evals API.
    """

    _MANAGED_EVALS_BASE = "/api/2.0/managed-evals"
    _MANAGED_EVALS_SCORERS_BASE = f"{_MANAGED_EVALS_BASE}/scheduled-scorers"

    def __init__(self, tracking_uri=None):
        self.get_host_creds = partial(get_databricks_host_creds, tracking_uri)

    @staticmethod
    def _encode_path_param(value: str) -> str:
        return quote(str(value), safe="")

    @staticmethod
    def _scorer_resource_key(name: str) -> str:
        return urlsafe_b64encode(name.encode("utf-8")).decode("ascii").rstrip("=")

    def _scheduled_scorers_endpoint(self, experiment_id: str) -> str:
        return f"{self._MANAGED_EVALS_SCORERS_BASE}/{self._encode_path_param(experiment_id)}"

    def _scheduled_scorer_resource_parent(self, experiment_id: str, name: str) -> str:
        return (
            f"experiments/{self._encode_path_param(experiment_id)}"
            f"/scheduledScorers/{self._scorer_resource_key(name)}"
        )

    @staticmethod
    def _validate_version(version: Any) -> int:
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise MlflowException.invalid_parameter_value(
                f"`version` must be a positive integer, got {version!r}."
            )
        return version

    def _scheduled_scorer_version_resource_name(
        self, experiment_id: str, name: str, version: int
    ) -> str:
        version = self._validate_version(version)
        parent = self._scheduled_scorer_resource_parent(experiment_id, name)
        return f"{parent}/versions/{version}"

    def _scorer_version_endpoint(self, experiment_id: str, name: str, version: int) -> str:
        resource_name = self._scheduled_scorer_version_resource_name(experiment_id, name, version)
        return f"{self._MANAGED_EVALS_BASE}/{resource_name}"

    def _scorer_versions_endpoint(self, experiment_id: str, name: str) -> str:
        parent = self._scheduled_scorer_resource_parent(experiment_id, name)
        return f"{self._MANAGED_EVALS_BASE}/{parent}/versions"

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json=json_body,
            params=params,
        )
        verify_rest_response(response, endpoint)
        if not response.text:
            return {}
        return response.json()

    @staticmethod
    def _extract_current_scorer_configs(response: dict[str, Any]) -> list[dict[str, Any]]:
        scheduled_scorers = response.get("scheduled_scorers", {})
        if "scorers" in scheduled_scorers:
            return scheduled_scorers["scorers"]
        return scheduled_scorers.get("scorers_config", {}).get("scorers", [])

    @staticmethod
    def _get_config_version(config: dict[str, Any]) -> int:
        return config.get("version", 1)

    @staticmethod
    def _scorer_type_fields(serialized: dict[str, Any]) -> dict[str, Any]:
        if serialized.get("builtin_scorer_class"):
            builtin_data = serialized.get("builtin_scorer_pydantic_data")
            builtin_name = (
                builtin_data.get("name") if isinstance(builtin_data, dict) else None
            ) or serialized.get("name")
            return {"builtin": {"name": builtin_name}}

        return {"custom": {}}

    @classmethod
    def _scorer_config(
        cls,
        *,
        name: str,
        scorer: Scorer,
        sample_rate: float | None,
        filter_string: str | None,
        version: int | None = None,
    ) -> dict[str, Any]:
        serialized_scorer = scorer.model_dump()
        config: dict[str, Any] = {
            "name": name,
            "serialized_scorer": json.dumps(serialized_scorer),
            **cls._scorer_type_fields(serialized_scorer),
        }
        if sample_rate is not None:
            config["sample_rate"] = sample_rate
        if filter_string is not None:
            config["filter_string"] = filter_string
        if version is not None:
            config["version"] = version
        return config

    def _list_current_scorer_configs(self, experiment_id: str) -> list[dict[str, Any]]:
        response = self._request("GET", self._scheduled_scorers_endpoint(experiment_id))
        return self._extract_current_scorer_configs(response)

    def _patch_current_scorer_configs(
        self, experiment_id: str, configs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        response = self._request(
            "PATCH",
            self._scheduled_scorers_endpoint(experiment_id),
            json_body={
                "scheduled_scorers": {"scorers": configs},
                "update_mask": "scheduled_scorers.scorers",
            },
        )
        updated_configs = self._extract_current_scorer_configs(response)
        return updated_configs or configs

    def _hydrate_scorer(
        self,
        scorer: Scorer,
        experiment_id: str,
        *,
        sample_rate: float | None = None,
        filter_string: str | None = None,
        version: int | None = None,
    ) -> Scorer:
        scorer._registered_backend = SCORER_BACKEND_DATABRICKS
        scorer._experiment_id = experiment_id
        scorer._sampling_config = ScorerSamplingConfig(
            sample_rate=sample_rate,
            filter_string=filter_string,
        )
        scorer._registered_scorer_version = version
        return scorer

    def _config_to_scorer(
        self,
        config: dict[str, Any],
        experiment_id: str,
        *,
        current_config: dict[str, Any] | None = None,
        display_name: str | None = None,
    ) -> Scorer:
        serialized_scorer = config.get("serialized_scorer")
        if serialized_scorer is None:
            raise MlflowException.invalid_parameter_value(
                "Scheduled scorer response did not include `serialized_scorer`."
            )
        scorer = Scorer.model_validate(json.loads(serialized_scorer))
        display_name = display_name or config.get("name")
        if display_name is not None and scorer.name != display_name:
            scorer.name = display_name
            if scorer._cached_dump is not None:
                scorer._cached_dump["name"] = display_name
        sampling_config = current_config if current_config is not None else config
        return self._hydrate_scorer(
            scorer,
            experiment_id,
            sample_rate=sampling_config.get("sample_rate"),
            filter_string=sampling_config.get("filter_string"),
            version=self._get_config_version(config),
        )

    def _find_current_scorer_config(self, experiment_id: str, name: str) -> dict[str, Any]:
        for config in self._list_current_scorer_configs(experiment_id):
            if config.get("name") == name:
                return config
        raise MlflowException(
            f"Scorer with name '{name}' not found for experiment {experiment_id}."
        )

    def _add_or_update_registered_scorer(
        self,
        *,
        name: str,
        scorer: Scorer,
        sample_rate: float | None,
        filter_string: str | None,
        experiment_id: str | None,
    ) -> Scorer:
        experiment_id = experiment_id or _get_experiment_id()
        configs = self._list_current_scorer_configs(experiment_id)
        updated_config = self._scorer_config(
            name=name,
            scorer=scorer,
            sample_rate=sample_rate,
            filter_string=filter_string,
        )
        replaced = False
        for i, config in enumerate(configs):
            if config.get("name") == name:
                for key in ("sample_rate", "filter_string"):
                    if key in config:
                        updated_config[key] = config[key]
                    else:
                        updated_config.pop(key, None)
                updated_config["version"] = self._get_config_version(config)
                configs[i] = updated_config
                replaced = True
                break
        if not replaced:
            configs.append(updated_config)

        updated_configs = self._patch_current_scorer_configs(experiment_id, configs)
        for config in updated_configs:
            if config.get("name") == name:
                return self._config_to_scorer(config, experiment_id)
        return self._config_to_scorer(updated_config, experiment_id)

    def _update_registered_scorer_schedule(
        self,
        *,
        name: str,
        sample_rate: float | None,
        filter_string: str | None,
        experiment_id: str,
    ) -> Scorer:
        configs = self._list_current_scorer_configs(experiment_id)
        updated_config = None
        for i, config in enumerate(configs):
            if config.get("name") == name:
                updated_config = dict(config)
                if sample_rate is not None:
                    updated_config["sample_rate"] = sample_rate
                if filter_string is not None:
                    updated_config["filter_string"] = filter_string
                configs[i] = updated_config
                break

        if updated_config is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}."
            )

        updated_configs = self._patch_current_scorer_configs(experiment_id, configs)
        for config in updated_configs:
            if config.get("name") == name:
                return self._config_to_scorer(config, experiment_id)
        return self._config_to_scorer(updated_config, experiment_id)

    def _delete_current_scorer(self, experiment_id: str | None, name: str) -> None:
        experiment_id = experiment_id or _get_experiment_id()
        current_configs = self._list_current_scorer_configs(experiment_id)
        configs = [config for config in current_configs if config.get("name") != name]
        if len(configs) == len(current_configs):
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}."
            )
        self._patch_current_scorer_configs(experiment_id, configs)

    # Private functions for internal use by Scorer methods
    def add_registered_scorer(
        self,
        *,
        name: str,
        scorer: Scorer,
        sample_rate: float,
        filter_string: str | None = None,
        experiment_id: str | None = None,
    ) -> Scorer:
        """Internal function to add a registered scorer."""
        return self._add_or_update_registered_scorer(
            name=name,
            scorer=scorer,
            sample_rate=sample_rate,
            filter_string=filter_string,
            experiment_id=experiment_id,
        )

    @staticmethod
    def list_scheduled_scorers(experiment_id):
        store = _get_scorer_store()
        experiment_id = experiment_id or _get_experiment_id()
        return [
            store._config_to_scorer(config, experiment_id)
            for config in store._list_current_scorer_configs(experiment_id)
        ]

    @staticmethod
    def get_scheduled_scorer(name, experiment_id):
        store = _get_scorer_store()
        experiment_id = experiment_id or _get_experiment_id()
        return store._config_to_scorer(
            store._find_current_scorer_config(experiment_id, name),
            experiment_id,
        )

    @staticmethod
    def delete_scheduled_scorer(experiment_id, name):
        _get_scorer_store()._delete_current_scorer(experiment_id, name)

    def update_registered_scorer(
        self,
        *,
        name: str,
        scorer: Scorer | None = None,
        sample_rate: float | None = None,
        filter_string: str | None = None,
        experiment_id: str | None = None,
    ) -> Scorer:
        """Update scheduling fields without changing the current scorer definition."""
        experiment_id = (
            experiment_id
            or (scorer._experiment_id if scorer is not None else None)
            or _get_experiment_id()
        )
        return self._update_registered_scorer_schedule(
            name=name,
            sample_rate=sample_rate,
            filter_string=filter_string,
            experiment_id=experiment_id,
        )

    def register_scorer(self, experiment_id: str | None, scorer: Scorer) -> int | None:
        registered_scorer = self.add_registered_scorer(
            name=scorer.name,
            scorer=scorer,
            sample_rate=0.0,
            filter_string=None,
            experiment_id=experiment_id,
        )

        scorer._registered_backend = SCORER_BACKEND_DATABRICKS
        if isinstance(registered_scorer, Scorer):
            scorer._experiment_id = registered_scorer._experiment_id
            scorer._sampling_config = ScorerSamplingConfig(
                sample_rate=registered_scorer.sample_rate,
                filter_string=registered_scorer.filter_string,
            )
            version = registered_scorer._registered_scorer_version
        else:
            scorer._experiment_id = experiment_id
            scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.0)
            version = None
        scorer._registered_scorer_version = version
        return version

    def list_scorers(self, experiment_id) -> list["Scorer"]:
        experiment_id = experiment_id or _get_experiment_id()
        return [
            self._config_to_scorer(config, experiment_id)
            for config in self._list_current_scorer_configs(experiment_id)
        ]

    def get_scorer(self, experiment_id, name, version=None) -> "Scorer":
        if version is not None:
            experiment_id = experiment_id or _get_experiment_id()
            response = self._request(
                "GET",
                self._scorer_version_endpoint(experiment_id, name, version),
            )
            current_config = self._find_current_scorer_config(experiment_id, name)
            return self._config_to_scorer(
                response,
                experiment_id,
                current_config=current_config,
                display_name=response["display_name"],
            )

        experiment_id = experiment_id or _get_experiment_id()
        return self._config_to_scorer(
            self._find_current_scorer_config(experiment_id, name),
            experiment_id,
        )

    def list_scorer_versions(self, experiment_id, name) -> list[tuple["Scorer", int]]:
        experiment_id = experiment_id or _get_experiment_id()
        endpoint = self._scorer_versions_endpoint(experiment_id, name)
        configs = []
        page_token = None
        while True:
            params = {}
            if page_token:
                params["page_token"] = page_token
            response = self._request("GET", endpoint, params=params or None)
            configs.extend(response.get("scheduled_scorer_versions", []))
            page_token = response.get("next_page_token")
            if not page_token:
                break

        current_config = self._find_current_scorer_config(experiment_id, name)
        return [
            (
                self._config_to_scorer(
                    config,
                    experiment_id,
                    current_config=current_config,
                    display_name=config["display_name"],
                ),
                self._get_config_version(config),
            )
            for config in configs
        ]

    def delete_scorer(self, experiment_id, name, version):
        if version is None or version == "all":
            return self._delete_current_scorer(experiment_id, name)

        experiment_id = experiment_id or _get_experiment_id()
        self._request(
            "DELETE",
            self._scorer_version_endpoint(experiment_id, name, version),
        )


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


def list_scorer_versions(
    *, name: str, experiment_id: str | None = None
) -> list[tuple[Scorer, int]]:
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
        list[tuple[Scorer, int]]: A list of tuples, where each tuple contains:
            - A Scorer object representing the scorer at that specific version
            - An integer representing the version number (1, 2, 3, etc.).
            The list may be empty if no versions of the scorer exist.

    Raises:
        mlflow.MlflowException: If the scorer with the specified name is not found in
            the experiment, if the experiment doesn't exist, or if there are issues with the backend
            store.
    """
    store = _get_scorer_store()
    return store.list_scorer_versions(experiment_id, name)


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
    """

    store = _get_scorer_store()
    return store.get_scorer(experiment_id, name, version)


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
        - Supports deleting a specific version or all versions
        - For backwards compatibility, `version=None` deletes all versions

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
            For Databricks backend, `None` is also accepted as a backwards-compatible
            spelling for deleting all versions

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
