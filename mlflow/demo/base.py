import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

DEMO_EXPERIMENT_NAME = "MLflow Demo"
DEMO_PROMPT_PREFIX = "mlflow-demo"


class DemoFeature(str, Enum):
    """Enumeration of demo features that can be generated."""

    TRACES = "traces"
    EVALUATION = "evaluation"
    PROMPTS = "prompts"


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
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
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
        if experiment := store.get_experiment_by_name(DEMO_EXPERIMENT_NAME):
            tag = ExperimentTag(
                key=f"mlflow.demo.version.{self.name}",
                value=str(self.version),
            )
            store.set_experiment_tag(experiment.experiment_id, tag)
