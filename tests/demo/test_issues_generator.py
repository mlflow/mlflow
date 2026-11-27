import pytest

import mlflow
from mlflow.demo.base import DEMO_EXPERIMENT_NAME, DemoFeature, DemoResult
from mlflow.demo.generators.issues import IssuesDemoGenerator
from mlflow.demo.generators.traces import DEMO_VERSION_TAG
from mlflow.entities import SpanType
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.issue import IssueStatus
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_ISSUE_DETECTION,
)


@pytest.fixture
def restore_version():
    """The version gate is a class attribute, so a test that moves it has to put it back."""
    original = IssuesDemoGenerator.version
    yield
    IssuesDemoGenerator.version = original


@pytest.fixture
def judged_trace():
    """One trace the generator will file an issue from: v1, with a failing judge."""
    mlflow.set_experiment(DEMO_EXPERIMENT_NAME)
    root = mlflow.start_span_no_context(
        name="rag_pipeline",
        span_type=SpanType.CHAIN,
        inputs={"question": "question"},
        metadata={DEMO_VERSION_TAG: "v1"},
    )
    llm = mlflow.start_span_no_context(
        name="generate_answer", span_type=SpanType.LLM, parent_span=root
    )
    llm.end()
    root.end()

    mlflow.log_feedback(
        trace_id=root.trace_id,
        name="groundedness",
        value="no",
        rationale="The answer asserts a figure the retrieved context does not contain.",
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id="judges/groundedness"
        ),
    )
    return mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME).experiment_id


def test_generator_attributes():
    generator = IssuesDemoGenerator()

    assert generator.name == DemoFeature.ISSUES


def test_data_exists_is_false_before_anything_is_generated():
    assert IssuesDemoGenerator()._data_exists() is False


def test_generate_reports_what_it_made(judged_trace):
    experiment_id = judged_trace

    result = IssuesDemoGenerator().generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.ISSUES
    # The ids are what the caller offers to navigate to, so an empty list would mean the demo
    # generated nothing while reporting success.
    assert result.entity_ids
    assert result.navigation_url == f"#/experiments/{experiment_id}/issues"


def test_data_exists_is_true_once_an_issue_has_been_filed(judged_trace):
    generator = IssuesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_clears_what_a_regeneration_would_otherwise_duplicate(judged_trace):
    experiment_id = judged_trace
    generator = IssuesDemoGenerator()
    generator.generate()

    generator.delete_demo()

    store = _get_store()
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
        max_results=100,
    )
    assert runs.empty
    # Rejected rather than removed, because there is no delete-issue API: the default view shows
    # active issues, so regenerating puts fresh ones on top instead of duplicating names. This is
    # also why `_data_exists` stays true here -- it counts issues of any status.
    assert all(
        issue.status == IssueStatus.REJECTED
        for issue in store.search_issues(experiment_id=experiment_id)
    )
    # Both, or a stale trace filter outlives the demo and narrows the next one's runs.
    tags = store.get_experiment(experiment_id).tags
    assert MLFLOW_ISSUE_DETECTION_SCHEDULE not in tags
    assert MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER not in tags


def test_is_generated_turns_false_when_the_generator_outgrows_the_stored_version(
    judged_trace, restore_version
):
    generator = IssuesDemoGenerator()
    generator.generate()
    generator.store_version()
    assert generator.is_generated() is True

    # What a version bump has to achieve: the old demo is not mistaken for the current one.
    IssuesDemoGenerator.version = IssuesDemoGenerator.version + 1

    assert IssuesDemoGenerator().is_generated() is False
