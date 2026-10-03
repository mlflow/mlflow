import json

import pytest

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.mcp.tools._types import ExperimentPage
from mlflow.mcp.tools.experiments import (
    create_experiment,
    delete_experiment,
    get_experiment,
    rename_experiment,
    restore_experiment,
    search_experiments,
    update_experiment,
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracing.constant import TraceExperimentTagKey
from mlflow.tracking._tracking_service.utils import _get_store


def test_create_and_get_experiment_by_id_and_name():
    created = create_experiment("exp-a", trace_archival_retention="30d")
    assert created.name == "exp-a"

    by_id = get_experiment(experiment_id=created.experiment_id)
    by_name = get_experiment(experiment_name="exp-a")
    assert by_id == by_name
    assert by_id.lifecycle_stage == "active"
    assert json.loads(by_id.tags[TraceExperimentTagKey.ARCHIVAL_RETENTION]) == {
        "type": "duration",
        "value": "30d",
    }


def test_create_experiment_validates_retention():
    with pytest.raises(MlflowException, match="retention"):
        create_experiment("exp-a", trace_archival_retention="forever")


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"experiment_id": "0", "experiment_name": "Default"}],
)
def test_get_experiment_requires_exactly_one_identifier(kwargs):
    with pytest.raises(MlflowException, match="exactly one of experiment_id or experiment_name"):
        get_experiment(**kwargs)


def test_get_experiment_by_missing_name():
    with pytest.raises(MlflowException, match="'missing' does not exist"):
        get_experiment(experiment_name="missing")


def test_update_experiment_sets_and_clears_archival_controls():
    experiment_id = create_experiment("exp-a").experiment_id

    result = update_experiment(
        experiment_id, trace_archival_retention="12h", trace_archive_now=True
    )
    assert result.changes == [
        "set trace archival retention to 12h",
        "requested archive-now on the next scheduler pass",
    ]
    tags = get_experiment(experiment_id=experiment_id).tags
    assert TraceExperimentTagKey.ARCHIVE_NOW in tags

    result = update_experiment(
        experiment_id, clear_trace_archival_retention=True, clear_trace_archive_now=True
    )
    assert result.changes == [
        "cleared trace archival retention override",
        "cleared pending archive-now request",
    ]
    tags = get_experiment(experiment_id=experiment_id).tags
    assert TraceExperimentTagKey.ARCHIVAL_RETENTION not in tags
    assert TraceExperimentTagKey.ARCHIVE_NOW not in tags


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "at least one update option"),
        ({"trace_archival_retention": "1d", "clear_trace_archival_retention": True}, "both"),
        ({"trace_archive_now": True, "trace_archive_now_older_than": "1d"}, "both"),
        ({"trace_archive_now": True, "clear_trace_archive_now": True}, "together"),
        ({"trace_archive_now_older_than": "soon"}, "retention"),
    ],
)
def test_update_experiment_rejects_invalid_combinations(kwargs, match):
    experiment_id = create_experiment("exp-a").experiment_id
    with pytest.raises(MlflowException, match=match):
        update_experiment(experiment_id, **kwargs)


def test_search_experiments_returns_one_default_page_and_a_token():
    store = _get_store()
    created = {store.create_experiment(f"exp-{i}") for i in range(SEARCH_MAX_RESULTS_DEFAULT + 1)}

    first = search_experiments()
    assert len(first.experiments) == SEARCH_MAX_RESULTS_DEFAULT
    assert first.next_page_token is not None

    seen = [e.experiment_id for e in first.experiments]
    page_token = first.next_page_token
    while page_token:
        page = search_experiments(page_token=page_token)
        seen.extend(e.experiment_id for e in page.experiments)
        page_token = page.next_page_token
    assert len(seen) == len(set(seen)) == len(created) + 1
    assert set(seen) == created | {"0"}


def test_search_experiments_pages_with_the_store_token():
    for i in range(5):
        create_experiment(f"exp-{i}")

    names = []
    page_token = None
    while True:
        page = search_experiments(max_results=2, page_token=page_token, order_by=["name ASC"])
        assert len(page.experiments) <= 2
        names.extend(e.name for e in page.experiments)
        if not (page_token := page.next_page_token):
            break
    assert names == ["Default", "exp-0", "exp-1", "exp-2", "exp-3", "exp-4"]


def test_search_experiments_filters_and_orders():
    create_experiment("prod-a")
    create_experiment("prod-b")
    create_experiment("dev-a")
    page = search_experiments(filter_string="name LIKE 'prod-%'", order_by="name DESC")
    assert [e.name for e in page.experiments] == ["prod-b", "prod-a"]


def test_search_experiments_view_and_lifecycle_tools():
    experiment_id = create_experiment("exp-a").experiment_id
    assert delete_experiment(experiment_id).experiment_id == experiment_id
    assert [e.name for e in search_experiments(view="deleted_only").experiments] == ["exp-a"]

    restore_experiment(experiment_id)
    assert search_experiments(view="deleted_only").experiments == []

    renamed = rename_experiment(experiment_id, "exp-b")
    assert renamed.name == "exp-b"
    assert MlflowClient().get_experiment(experiment_id).name == "exp-b"


def test_search_experiments_rejects_negative_max_results():
    with pytest.raises(MlflowException, match="non-negative"):
        search_experiments(max_results=-1)


def test_search_experiments_returns_an_empty_page_for_zero_max_results():
    create_experiment("exp-a")
    assert search_experiments(max_results=0) == ExperimentPage(experiments=[], next_page_token=None)
    # Nothing is consumed, so the page resumes where it started.
    first = search_experiments(max_results=1)
    assert first.next_page_token is not None
    resumed = search_experiments(max_results=0, page_token=first.next_page_token)
    assert resumed == ExperimentPage(experiments=[], next_page_token=first.next_page_token)
