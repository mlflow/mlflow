import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

DEMO_EXPERIMENT_NAME = "MLflow Demo"
DEMO_PROMPT_PREFIX = "mlflow-demo"

# Unity Catalog schema for prompts/judges (e.g. "catalog.schema").
# Set via set_uc_schema() before demo generation when targeting Databricks.
_uc_schema: str | None = None


def set_uc_schema(schema: str | None) -> None:
    global _uc_schema
    _uc_schema = schema


def resolve_demo_name(name: str) -> str:
    """Resolve a demo prompt/judge name for the current backend.

    For Unity Catalog, replaces the default ``DEMO_PROMPT_PREFIX.*`` prefix
    with the configured UC ``catalog.schema``, prefixing the short name with
    ``mlflow_demo_`` so that demo entities remain distinguishable in a shared schema.

    Example::

        resolve_demo_name("mlflow-demo.prompts.customer_support")
        # local  -> "mlflow-demo.prompts.customer_support"
        # UC     -> "my_catalog.my_schema.mlflow_demo_customer_support"
    """
    if _uc_schema is None:
        return name
    short_name = name.rsplit(".", 1)[-1]
    return f"{_uc_schema}.mlflow_demo_{short_name}"


def is_demo_prompt_name(name: str) -> bool:
    """Check whether *name* belongs to a demo prompt."""
    if _uc_schema is not None:
        short_name = name.rsplit(".", 1)[-1]
        return short_name.startswith("mlflow_demo_")
    return name.startswith(f"{DEMO_PROMPT_PREFIX}.")


def get_demo_prompt_search_filter() -> str:
    """Return the appropriate search filter for finding demo prompts."""
    if _uc_schema is not None:
        match _uc_schema.split("."):
            case [catalog, schema]:
                return (
                    f"catalog = '{catalog}' AND schema = '{schema}' AND name LIKE 'mlflow_demo_%'"
                )
            case _:
                return f"name LIKE '{DEMO_PROMPT_PREFIX}.%'"
    return f"name LIKE '{DEMO_PROMPT_PREFIX}.%'"


@functools.cache
def get_demo_experiment_name() -> str:
    """Return the demo experiment name, adapting for Databricks workspaces.

    On Databricks, experiment names must be absolute paths
    (e.g., ``/Users/<username>/MLflow Demo``). For local or remote MLflow
    servers the plain name ``MLflow Demo`` is returned unchanged.

    The result is cached for the lifetime of the process.
    """
    import mlflow
    from mlflow.utils.uri import is_databricks_uri

    tracking_uri = mlflow.get_tracking_uri()
    if is_databricks_uri(tracking_uri):
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        username = w.current_user.me().user_name
        return f"/Users/{username}/{DEMO_EXPERIMENT_NAME}"
    return DEMO_EXPERIMENT_NAME


class DemoFeature(str, Enum):
    """Enumeration of demo features that can be generated."""

    TRACES = "traces"
    EVALUATION = "evaluation"
    PROMPTS = "prompts"
    JUDGES = "judges"


@dataclass
class DemoResult:
    """Result returned by a demo generator after creating demo data.

    Attributes:
        feature: The demo feature that was generated. Use DemoFeature enum values.
        entity_ids: List of identifiers for created entities (e.g., trace IDs, dataset names).
        navigation_url: URL path to navigate to view the demo data in the UI.
    """

    feature: DemoFeature
    entity_ids: list[str]
    navigation_url: str


class BaseDemoGenerator(ABC):
    """Abstract base class for demo data generators.

    Subclasses must define a `name` class attribute and implement the `generate()`
    and `_data_exists()` methods. Generators are registered with the `demo_registry`
    and invoked during server startup to populate demo data.

    Versioning:
        Each generator has a `version` class attribute (default: 1). When demo data
        is generated, the version is stored as a tag on the MLflow Demo experiment.
        On subsequent startups, if the stored version doesn't match the generator's
        current version, stale data is cleaned up and regenerated.

        Bump the version when making breaking changes to demo data format.

    Example:
        class MyDemoGenerator(BaseDemoGenerator):
            name = DemoFeature.TRACES
            version = 1  # Bump when demo format changes

            def generate(self) -> DemoResult:
                # Create demo data using MLflow APIs
                return DemoResult(...)

            def _data_exists(self) -> bool:
                # Check if demo data exists (version handled by base class)
                return True/False

            def delete_demo(self) -> None:
                # Optional: delete demo data (called on version mismatch or via UI)
                pass
    """

    name: DemoFeature | None = None
    version: int = 1

    def __init__(self):
        if self.name is None:
            raise ValueError(f"{self.__class__.__name__} must define 'name' class attribute")

    @abstractmethod
    def generate(self) -> DemoResult:
        """Generate demo data for this feature. Returns a DemoResult with details."""

    @abstractmethod
    def _data_exists(self) -> bool:
        """Check if demo data exists (regardless of version)."""

    def delete_demo(self) -> None:
        """Delete demo data created by this generator.

        Called automatically when version mismatches on startup, or can be called
        directly via API for user-initiated deletion. Override to implement cleanup.
        """

    def is_generated(self) -> bool:
        """Check if demo data exists with a matching version.

        Returns True only if data exists AND the stored version matches the current
        generator version. If version mismatches, calls delete_demo() and
        returns False to trigger regeneration.
        """
        if not self._data_exists():
            return False

        stored_version = self._get_stored_version()
        if stored_version is None or stored_version != self.version:
            self.delete_demo()
            return False

        return True

    def _get_stored_version(self) -> int | None:
        """Get the stored version for this generator from experiment tags."""
        store = _get_store()
        try:
            experiment = store.get_experiment_by_name(get_demo_experiment_name())
            if experiment is None:
                return None
            version_tag = experiment.tags.get(f"mlflow.demo.version.{self.name}")
            return int(version_tag) if version_tag else None
        except Exception:
            _logger.debug("Failed to get stored version for %s", self.name, exc_info=True)
            return None

    def store_version(self) -> None:
        """Store the current version in experiment tags. Called after successful generation."""
        from mlflow.entities import ExperimentTag

        store = _get_store()
        if experiment := store.get_experiment_by_name(get_demo_experiment_name()):
            tag = ExperimentTag(
                key=f"mlflow.demo.version.{self.name}",
                value=str(self.version),
            )
            store.set_experiment_tag(experiment.experiment_id, tag)
