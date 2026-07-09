import pytest

import mlflow
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.review_queues import (
    DEMO_DEFAULT_REVIEWER,
    DEMO_LABEL_SCHEMAS,
    DEMO_REVIEW_QUEUE_NAME,
    DEMO_REVIEWERS,
    ReviewQueuesDemoGenerator,
)
from mlflow.tracking._tracking_service.utils import _get_store


@pytest.fixture
def demo_experiment_with_traces():
    """Seed the demo experiment with a handful of traces for queues to attach to."""
    mlflow.set_experiment(DEMO_EXPERIMENT_NAME)

    @mlflow.trace
    def predict(question: str) -> str:
        return f"answer to {question}"

    for i in range(15):
        predict(f"question {i}")

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    return experiment.experiment_id


@pytest.fixture
def review_queues_generator():
    generator = ReviewQueuesDemoGenerator()
    original_version = generator.version
    yield generator
    ReviewQueuesDemoGenerator.version = original_version


def test_generator_attributes():
    generator = ReviewQueuesDemoGenerator()
    assert generator.name == DemoFeature.REVIEW_QUEUES
    assert generator.version == 1


def test_data_exists_false_when_no_queues():
    generator = ReviewQueuesDemoGenerator()
    assert generator._data_exists() is False


def test_generate_creates_queues(demo_experiment_with_traces):
    generator = ReviewQueuesDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.REVIEW_QUEUES
    assert "/review-queue" in result.navigation_url
    # One custom queue plus one user queue per reviewer and the default (viewer) queue.
    assert len(result.entity_ids) == 1 + len(DEMO_REVIEWERS) + 1


def test_generate_creates_schemas_queue_and_items(demo_experiment_with_traces):
    experiment_id = demo_experiment_with_traces
    generator = ReviewQueuesDemoGenerator()
    generator.generate()

    store = _get_store()

    schema_names = {s.name for s in store.list_label_schemas(experiment_id, max_results=100)}
    for spec in DEMO_LABEL_SCHEMAS:
        assert spec["name"] in schema_names

    custom_queue = store.get_review_queue_by_name(experiment_id, name=DEMO_REVIEW_QUEUE_NAME)
    assert custom_queue.queue_type == "custom"
    assert set(custom_queue.users) == set(DEMO_REVIEWERS)
    assert len(custom_queue.schema_ids) == len(DEMO_LABEL_SCHEMAS)

    items = store.list_review_queue_items(custom_queue.queue_id)
    assert len(items) > 0
    # Progress seeding marks some items as terminal (complete/declined).
    assert any(item.status != "pending" for item in items)

    for reviewer in [*DEMO_REVIEWERS, DEMO_DEFAULT_REVIEWER]:
        user_queue = store.get_review_queue_by_name(experiment_id, name=reviewer)
        assert user_queue.queue_type == "user"
        assert len(store.list_review_queue_items(user_queue.queue_id)) > 0


def test_data_exists_true_after_generate(demo_experiment_with_traces):
    generator = ReviewQueuesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_queues(demo_experiment_with_traces):
    generator = ReviewQueuesDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False


def test_generate_is_idempotent_after_partial_schema_creation(demo_experiment_with_traces):
    experiment_id = demo_experiment_with_traces
    store = _get_store()
    # Simulate a prior run that created the schemas but not the queue (which
    # `_data_exists` keys off), so a re-run must not crash on duplicate schema names.
    for spec in DEMO_LABEL_SCHEMAS:
        store.create_label_schema(
            experiment_id,
            name=spec["name"],
            type=spec["type"],
            input=spec["input"],
            instruction=spec["instruction"],
            enable_comment=spec["enable_comment"],
        )

    generator = ReviewQueuesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_generate_without_traces_still_creates_queues():
    mlflow.set_experiment(DEMO_EXPERIMENT_NAME)
    generator = ReviewQueuesDemoGenerator()

    result = generator.generate()

    assert generator._data_exists() is True
    assert len(result.entity_ids) == 1 + len(DEMO_REVIEWERS) + 1


def test_is_generated_checks_version(demo_experiment_with_traces, review_queues_generator):
    review_queues_generator.generate()
    review_queues_generator.store_version()

    assert review_queues_generator.is_generated() is True

    ReviewQueuesDemoGenerator.version = 99
    fresh_generator = ReviewQueuesDemoGenerator()
    assert fresh_generator.is_generated() is False
