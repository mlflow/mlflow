from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from kubernetes import client, config, watch
from kubernetes.client import CoreV1Api
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.workspace.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTION_ANNOTATION = "mlflow.kubeflow.org/workspace-description"
DEFAULT_NAMESPACE_EXCLUDE_GLOBS: tuple[str, ...] = ("kube-*", "openshift-*")


def _parse_glob_input(value: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None

    candidates = value.split(",") if isinstance(value, str) else value
    return tuple(token.strip() for token in candidates if token and token.strip())


def _merge_globs(
    *glob_lists: tuple[str, ...] | None,
) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []

    for glob_list in glob_lists:
        if not glob_list:
            continue
        for pattern in glob_list:
            if pattern in seen:
                continue
            merged.append(pattern)
            seen.add(pattern)

    return tuple(merged)


@dataclass(frozen=True, slots=True)
class KubernetesWorkspaceProviderConfig:
    label_selector: str | None = None
    default_workspace: str | None = None
    namespace_exclude_globs: tuple[str, ...] = DEFAULT_NAMESPACE_EXCLUDE_GLOBS

    @classmethod
    def from_sources(
        cls,
        *,
        label_selector: str | None = None,
        default_workspace: str | None = None,
        namespace_exclude_globs: Iterable[str] | str | None = None,
    ) -> "KubernetesWorkspaceProviderConfig":
        env = os.environ

        env_globs = _parse_glob_input(env.get("MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS"))

        if namespace_exclude_globs is not None:
            glob_source = _parse_glob_input(namespace_exclude_globs) or ()
        elif env_globs is not None:
            glob_source = env_globs
        else:
            glob_source = ()

        exclude_globs = _merge_globs(DEFAULT_NAMESPACE_EXCLUDE_GLOBS, glob_source)

        return cls(
            label_selector=label_selector or env.get("MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR"),
            default_workspace=default_workspace or env.get("MLFLOW_K8S_DEFAULT_WORKSPACE"),
            namespace_exclude_globs=exclude_globs,
        )


@dataclass(frozen=True, slots=True)
class NamespaceInfo:
    name: str
    description: str | None


class NamespaceCache:
    """Caches namespace name/description pairs using a watch loop."""

    def __init__(
        self,
        api: CoreV1Api,
        label_selector: str | None,
        namespace_exclude_globs: tuple[str, ...],
    ):
        self._api = api
        self._label_selector = label_selector
        self._namespace_exclude_globs = namespace_exclude_globs
        self._lock = threading.RLock()
        self._namespaces: dict[str, NamespaceInfo] = {}
        self._resource_version: str | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._refresh_full()

        self._thread = threading.Thread(
            target=self._run,
            name="mlflow-k8s-namespace-watch",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def list_namespaces(self) -> list[NamespaceInfo]:
        self._wait_until_ready()
        with self._lock:
            return list(self._namespaces.values())

    def get_namespace(self, name: str) -> NamespaceInfo | None:
        self._wait_until_ready()
        with self._lock:
            return self._namespaces.get(name)

    def _wait_until_ready(self) -> None:
        # Namespace listings depend on a background watch loop that typically initializes within
        # a second. Provide a generous timeout so transient API hiccups (e.g., API server restarts)
        # do not immediately surface as user-facing errors.
        if self._ready_event.wait(timeout=30):
            return
        raise MlflowException(
            "Timed out while waiting for the Kubernetes namespace cache to initialize. "
            "Double-check the MLflow server can reach the Kubernetes API.",
            INTERNAL_ERROR,
        )

    def _refresh_full(self) -> None:
        try:
            response = self._api.list_namespace(label_selector=self._label_selector, watch=False)
        except ApiException as exc:  # pragma: no cover - depends on live cluster
            raise MlflowException(
                f"Failed to list Kubernetes namespaces: {exc}",
                INVALID_PARAMETER_VALUE,
            ) from exc

        items = getattr(response, "items", []) or []
        metadata = getattr(response, "metadata", None)
        resource_version = getattr(metadata, "resource_version", None)

        with self._lock:
            infos: dict[str, NamespaceInfo] = {}
            for ns in items:
                info = self._extract_info(ns)
                if info is not None:
                    infos[info.name] = info

            self._namespaces = infos
            self._resource_version = resource_version

        self._ready_event.set()

    def _run(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            if self._resource_version is None:
                try:
                    self._refresh_full()
                except MlflowException:
                    _logger.warning("Failed to refresh namespaces; retrying shortly", exc_info=True)
                    time.sleep(2)
                    continue

            watcher = watch.Watch()
            try:
                for event in watcher.stream(
                    self._api.list_namespace,
                    label_selector=self._label_selector,
                    resource_version=self._resource_version,
                    timeout_seconds=300,
                ):
                    if self._stop_event.is_set():
                        watcher.stop()
                        break
                    self._handle_event(event)
            except ApiException as exc:
                if exc.status == 410:
                    _logger.debug("Namespace watch resource version expired; resyncing.")
                    self._resource_version = None
                else:
                    _logger.warning("Namespace watch error: %s", exc)
                    time.sleep(2)
            except Exception:
                _logger.exception("Unexpected error in namespace watch loop")
                time.sleep(2)
            else:
                time.sleep(1)

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        obj = event.get("object")
        if not obj:
            return

        metadata = getattr(obj, "metadata", None)
        name = getattr(metadata, "name", None)
        if not name:
            return

        resource_version = getattr(metadata, "resource_version", None)

        with self._lock:
            if event_type in {"ADDED", "MODIFIED"}:
                info = self._extract_info(obj)
                if info is None:
                    self._namespaces.pop(name, None)
                else:
                    self._namespaces[name] = info
                self._resource_version = resource_version
            elif event_type == "DELETED":
                self._namespaces.pop(name, None)
                self._resource_version = resource_version
            elif event_type == "BOOKMARK":
                self._resource_version = resource_version or self._resource_version
            elif event_type == "ERROR":
                self._resource_version = None
                return
            else:
                # Unknown event type; trigger a resync.
                self._resource_version = None
                return

        self._ready_event.set()

    def _extract_info(self, namespace: object) -> NamespaceInfo | None:
        metadata = getattr(namespace, "metadata", None)
        name = getattr(metadata, "name", None)
        if not name:
            return None

        if self._is_excluded(name):
            return None

        annotations = getattr(metadata, "annotations", None) or {}
        description = annotations.get(DEFAULT_DESCRIPTION_ANNOTATION)

        return NamespaceInfo(name=name, description=description)

    def _is_excluded(self, name: str) -> bool:
        return any(fnmatchcase(name, pattern) for pattern in self._namespace_exclude_globs)


class KubernetesWorkspaceProvider(AbstractStore):
    """Workspace provider that maps MLflow workspaces to Kubernetes namespaces."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        label_selector: str | None = None,
        default_workspace: str | None = None,
        namespace_exclude_globs: Iterable[str] | str | None = None,
    ) -> None:
        del tracking_uri  # Unused but kept for compatibility with MLflow factory

        self._config = KubernetesWorkspaceProviderConfig.from_sources(
            label_selector=label_selector,
            default_workspace=default_workspace,
            namespace_exclude_globs=namespace_exclude_globs,
        )

        self._core_api = self._create_api_client(self._config)
        self._namespace_cache = NamespaceCache(
            self._core_api,
            self._config.label_selector,
            self._config.namespace_exclude_globs,
        )

    def list_workspaces(self) -> Iterable[Workspace]:  # type: ignore[override]
        infos = self._namespace_cache.list_namespaces()
        return [Workspace(name=info.name, description=info.description) for info in infos]

    def get_workspace(self, workspace_name: str) -> Workspace:  # type: ignore[override]
        info = self._namespace_cache.get_namespace(workspace_name)
        if info is None:
            parts = [
                f"Workspace '{workspace_name}' was not found in the Kubernetes cluster.",
                "Each MLflow workspace maps 1:1 to a namespace.",
            ]
            if self._config.label_selector:
                parts.append(
                    f"Ensure the namespace exists and matches the configured selector "
                    f"('{self._config.label_selector}')."
                )
            raise MlflowException(" ".join(parts), RESOURCE_DOES_NOT_EXIST)

        return Workspace(name=info.name, description=info.description)

    def create_workspace(self, workspace: Workspace) -> Workspace:  # type: ignore[override]
        raise NotImplementedError("Namespace creation is not supported by this provider")

    def update_workspace(self, workspace: Workspace) -> Workspace:  # type: ignore[override]
        raise NotImplementedError("Namespace updates are not supported by this provider")

    def delete_workspace(self, workspace_name: str) -> None:  # type: ignore[override]
        raise NotImplementedError("Namespace deletion is not supported by this provider")

    def get_default_workspace(self) -> Workspace:  # type: ignore[override]
        if not self._config.default_workspace:
            raise NotImplementedError(
                "Active workspace is required. Specify one in the request path or set "
                "MLFLOW_K8S_DEFAULT_WORKSPACE."
            )
        return self.get_workspace(self._config.default_workspace)

    @staticmethod
    def _create_api_client(config_values: KubernetesWorkspaceProviderConfig) -> CoreV1Api:
        try:
            config.load_incluster_config()
        except ConfigException:
            try:
                config.load_kube_config()
            except ConfigException as exc:  # pragma: no cover - depends on env
                raise MlflowException(
                    "Failed to load Kubernetes configuration for workspace provider",
                    error_code=INVALID_STATE,
                ) from exc
        return client.CoreV1Api()


def _parse_workspace_uri_options(
    workspace_uri: str | None,
) -> dict[str, str | tuple[str, ...] | None]:
    """
    Parse query parameters from the workspace URI to configure the provider.
    Supported parameters:
      - label_selector
      - default_workspace
      - namespace_exclude_globs (comma-separated glob patterns)
    """

    if not workspace_uri:
        return {}

    parsed = urlparse(workspace_uri)
    query = parse_qs(parsed.query, keep_blank_values=True)

    def _get_param(name: str) -> str | None:
        values = query.get(name) or []
        value = values[-1] if values else None
        if value is None:
            return None
        value = value.strip()
        return value or None

    options: dict[str, str | tuple[str, ...] | None] = {
        "label_selector": _get_param("label_selector"),
        "default_workspace": _get_param("default_workspace"),
    }

    exclude_param = _get_param("namespace_exclude_globs")
    if exclude_param is not None:
        parsed_excludes = _parse_glob_input(exclude_param) or ()
        options["namespace_exclude_globs"] = parsed_excludes
    else:
        options["namespace_exclude_globs"] = None

    return options


def create_kubernetes_workspace_store(workspace_uri: str, **_kwargs) -> KubernetesWorkspaceProvider:
    """
    Entry point factory that instantiates the Kubernetes workspace provider.

    Args:
        workspace_uri: The resolved workspace store URI. Query parameters are used
            to override provider configuration.
        _kwargs: Additional keyword arguments ignored for forward compatibility.
    """

    options = _parse_workspace_uri_options(workspace_uri)
    return KubernetesWorkspaceProvider(
        tracking_uri=workspace_uri,
        label_selector=options.get("label_selector"),
        default_workspace=options.get("default_workspace"),
        namespace_exclude_globs=options.get("namespace_exclude_globs"),
    )
