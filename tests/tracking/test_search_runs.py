from collections import namedtuple

import pytest

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking.client import MlflowClient

MockExperiment = namedtuple("MockExperiment", ["experiment_id", "lifecycle_stage"])


@pytest.mark.usefixtures("reset_active_experiment")
def test_search_runs():
    mlflow.set_experiment("exp-for-search")
    # Create a run and verify that the current active experiment is the one we just set
    logged_runs = {}
    with mlflow.start_run() as active_run:
        logged_runs["first"] = active_run.info.run_id
        mlflow.log_metric("m1", 0.001)
        mlflow.log_metric("m2", 0.002)
        mlflow.log_metric("m1", 0.002)
        mlflow.log_param("p1", "a")
        mlflow.set_tag("t1", "first-tag-val")
    with mlflow.start_run() as active_run:
        logged_runs["second"] = active_run.info.run_id
        mlflow.log_metric("m1", 0.008)
        mlflow.log_param("p2", "aa")
        mlflow.set_tag("t2", "second-tag-val")

    def verify_runs(runs, expected_set):
        assert set([r.info.run_id for r in runs]) == set([logged_runs[r] for r in expected_set])

    experiment_id = MlflowClient().get_experiment_by_name("exp-for-search").experiment_id

    # 2 runs in this experiment
    assert len(MlflowClient().list_run_infos(experiment_id, ViewType.ACTIVE_ONLY)) == 2

    # 2 runs that have metric "m1" > 0.001
    runs = MlflowClient().search_runs([experiment_id], "metrics.m1 > 0.0001")
    verify_runs(runs, ["first", "second"])

    # 1 run with has metric "m1" > 0.002
    runs = MlflowClient().search_runs([experiment_id], "metrics.m1 > 0.002")
    verify_runs(runs, ["second"])

    # no runs with metric "m1" > 0.1
    runs = MlflowClient().search_runs([experiment_id], "metrics.m1 > 0.1")
    verify_runs(runs, [])

    # 1 run with metric "m2" > 0
    runs = MlflowClient().search_runs([experiment_id], "metrics.m2 > 0")
    verify_runs(runs, ["first"])

    # 1 run each with param "p1" and "p2"
    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.ALL)
    verify_runs(runs, ["first"])
    runs = MlflowClient().search_runs([experiment_id], "params.p2 != 'a'", ViewType.ALL)
    verify_runs(runs, ["second"])
    runs = MlflowClient().search_runs([experiment_id], "params.p2 = 'aa'", ViewType.ALL)
    verify_runs(runs, ["second"])

    # 1 run each with tag "t1" and "t2"
    runs = MlflowClient().search_runs([experiment_id], "tags.t1 = 'first-tag-val'", ViewType.ALL)
    verify_runs(runs, ["first"])
    runs = MlflowClient().search_runs([experiment_id], "tags.t2 != 'qwerty'", ViewType.ALL)
    verify_runs(runs, ["second"])
    runs = MlflowClient().search_runs([experiment_id], "tags.t2 = 'second-tag-val'", ViewType.ALL)
    verify_runs(runs, ["second"])

    # delete "first" run
    MlflowClient().delete_run(logged_runs["first"])
    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.ALL)
    verify_runs(runs, ["first"])

    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.DELETED_ONLY)
    verify_runs(runs, ["first"])

    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.ACTIVE_ONLY)
    verify_runs(runs, [])


@pytest.mark.usefixtures("reset_active_experiment")
def test_search_runs_multiple_experiments():
    experiment_ids = [mlflow.create_experiment("exp__{}".format(exp_id)) for exp_id in range(1, 4)]
    for eid in experiment_ids:
        with mlflow.start_run(experiment_id=eid):
            mlflow.log_metric("m0", 1)
            mlflow.log_metric("m_{}".format(eid), 2)

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m0 > 0", ViewType.ALL)) == 3

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_1 > 0", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_2 = 2", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_3 < 4", ViewType.ALL)) == 1
