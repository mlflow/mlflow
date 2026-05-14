import time
from pathlib import Path

import pytest

from mlflow import entities
from mlflow.entities import (
    Experiment,
    ExperimentTag,
    ViewType,
)
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.tracking.dbmodels import models
from mlflow.store.tracking.dbmodels.models import SqlExperiment
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.file_utils import TempDir
from mlflow.utils.os import is_windows
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import MAX_EXPERIMENT_NAME_LENGTH

from tests.store.tracking.sqlalchemy_store.conftest import (
    _create_experiments,
    _get_run_configs,
    _run_factory,
)

pytestmark = pytest.mark.notrackingurimock


def test_default_experiment(store: SqlAlchemyStore):
    experiments = store.search_experiments()
    assert len(experiments) == 1

    first = experiments[0]
    assert first.experiment_id == "0"
    assert first.name == "Default"


def test_default_experiment_lifecycle(store: SqlAlchemyStore, tmp_path):
    default_experiment = store.get_experiment(experiment_id=0)
    assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    assert default_experiment.lifecycle_stage == entities.LifecycleStage.ACTIVE

    _create_experiments(store, "aNothEr")
    all_experiments = [e.name for e in store.search_experiments()]
    assert set(all_experiments) == {"aNothEr", "Default"}

    store.delete_experiment(0)

    assert [e.name for e in store.search_experiments()] == ["aNothEr"]
    another = store.get_experiment(1)
    assert another.name == "aNothEr"

    default_experiment = store.get_experiment(experiment_id=0)
    assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    assert default_experiment.lifecycle_stage == entities.LifecycleStage.DELETED

    # destroy SqlStore and make a new one
    db_uri = store.db_uri
    artifact_uri = store.artifact_root_uri
    del store
    store = SqlAlchemyStore(db_uri, artifact_uri)

    # test that default experiment is not reactivated
    default_experiment = store.get_experiment(experiment_id=0)
    assert default_experiment.name == Experiment.DEFAULT_EXPERIMENT_NAME
    assert default_experiment.lifecycle_stage == entities.LifecycleStage.DELETED

    assert [e.name for e in store.search_experiments()] == ["aNothEr"]
    all_experiments = [e.name for e in store.search_experiments(ViewType.ALL)]
    assert set(all_experiments) == {"aNothEr", "Default"}

    # ensure that experiment ID for active experiment is unchanged
    another = store.get_experiment(1)
    assert another.name == "aNothEr"

    if MLFLOW_TRACKING_URI.get():
        with store.ManagedSessionMaker() as session:
            default_exp = (
                session
                .query(SqlExperiment)
                .filter(SqlExperiment.experiment_id == store.DEFAULT_EXPERIMENT_ID)
                .first()
            )
            if default_exp:
                default_exp.lifecycle_stage = entities.LifecycleStage.ACTIVE
                session.commit()


def test_single_tenant_store_detects_workspace_scoped_experiments(
    tmp_path, db_uri, workspaces_enabled
):
    if workspaces_enabled:
        pytest.skip("Single-tenant startup guard only applies when workspaces are disabled.")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = SqlAlchemyStore(db_uri, artifact_dir.as_uri())
    exp_id = store.create_experiment("tenant-exp")
    with store.ManagedSessionMaker() as session:
        session.query(SqlExperiment).filter(SqlExperiment.experiment_id == exp_id).update({
            SqlExperiment.workspace: "another-workspace"
        })
        session.commit()
    store._dispose_engine()
    with pytest.raises(MlflowException, match="non-default workspaces"):
        SqlAlchemyStore(db_uri, artifact_dir.as_uri())


def test_artifact_path_segments_for_local():
    if is_windows():
        uri = "file:///C:/mlruns/workspaces/default"
        native_path = r"C:\mlruns\workspaces\default"
        expected = ["mlruns", "workspaces", "default"]
    else:
        uri = "file:///mlruns/workspaces/default"
        native_path = "/mlruns/workspaces/default"
        expected = ["mlruns", "workspaces", "default"]

    segments = SqlAlchemyStore._artifact_path_segments(uri)
    assert segments == expected

    segments_native = SqlAlchemyStore._artifact_path_segments(native_path)
    assert segments_native == expected


def test_raise_duplicate_experiments(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match=r"Experiment\(name=.+\) already exists"):
        _create_experiments(store, ["test", "test"])


def test_duplicate_experiment_with_artifact_location_returns_resource_already_exists(
    store: SqlAlchemyStore, tmp_path: Path, workspaces_enabled
):
    if workspaces_enabled:
        pytest.skip("Custom artifact locations are not supported when workspaces are enabled.")

    exp_name = "test_duplicate_with_artifact_location"
    artifact_location = str(tmp_path / "test_artifacts")

    # First creation should succeed
    store.create_experiment(exp_name, artifact_location=artifact_location)

    # Second creation should raise MlflowException with RESOURCE_ALREADY_EXISTS error code
    with pytest.raises(MlflowException, match="already exists") as exc_info:
        store.create_experiment(exp_name, artifact_location=artifact_location)

    # Verify that the error code is RESOURCE_ALREADY_EXISTS, not BAD_REQUEST
    assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_raise_experiment_dont_exist(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match=r"No Experiment with id=.+ exists"):
        store.get_experiment(experiment_id=100)


def test_delete_experiment(store: SqlAlchemyStore):
    experiments = _create_experiments(store, ["morty", "rick", "rick and morty"])

    all_experiments = store.search_experiments()
    assert len(all_experiments) == len(experiments) + 1  # default

    exp_id = experiments[0]
    exp = store.get_experiment(exp_id)
    time.sleep(0.01)
    store.delete_experiment(exp_id)

    updated_exp = store.get_experiment(exp_id)
    assert updated_exp.lifecycle_stage == entities.LifecycleStage.DELETED

    assert len(store.search_experiments()) == len(all_experiments) - 1
    assert updated_exp.last_update_time > exp.last_update_time


def test_delete_restore_experiment_with_runs(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "test exp")
    run1 = _run_factory(store, config=_get_run_configs(experiment_id)).info.run_id
    run2 = _run_factory(store, config=_get_run_configs(experiment_id)).info.run_id
    store.delete_run(run1)
    run_ids = [run1, run2]

    store.delete_experiment(experiment_id)

    updated_exp = store.get_experiment(experiment_id)
    assert updated_exp.lifecycle_stage == entities.LifecycleStage.DELETED

    deleted_run_list = store.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=ViewType.DELETED_ONLY,
    )

    assert len(deleted_run_list) == 2
    for deleted_run in deleted_run_list:
        assert deleted_run.info.lifecycle_stage == entities.LifecycleStage.DELETED
        assert deleted_run.info.experiment_id == experiment_id
        assert deleted_run.info.run_id in run_ids
        with store.ManagedSessionMaker() as session:
            assert store._get_run(session, deleted_run.info.run_id).deleted_time is not None

    store.restore_experiment(experiment_id)

    updated_exp = store.get_experiment(experiment_id)
    assert updated_exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

    restored_run_list = store.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    assert len(restored_run_list) == 2
    for restored_run in restored_run_list:
        assert restored_run.info.lifecycle_stage == entities.LifecycleStage.ACTIVE
        with store.ManagedSessionMaker() as session:
            assert store._get_run(session, restored_run.info.run_id).deleted_time is None
        assert restored_run.info.experiment_id == experiment_id
        assert restored_run.info.run_id in run_ids


def test_get_experiment(store: SqlAlchemyStore):
    name = "goku"
    experiment_id = _create_experiments(store, name)
    actual = store.get_experiment(experiment_id)
    assert actual.name == name
    assert actual.experiment_id == experiment_id

    actual_by_name = store.get_experiment_by_name(name)
    assert actual_by_name.name == name
    assert actual_by_name.experiment_id == experiment_id
    assert store.get_experiment_by_name("idontexist") is None

    store.delete_experiment(experiment_id)
    assert store.get_experiment_by_name(name).experiment_id == experiment_id


def test_get_experiment_invalid_id(store: SqlAlchemyStore):
    with pytest.raises(
        MlflowException,
        match=r"Invalid experiment ID 'invalid_id'\. Experiment ID must be a valid integer\.",
        check=lambda e: e.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE),
    ):
        store.get_experiment("invalid_id")


def test_search_experiments_view_type(store: SqlAlchemyStore):
    experiment_names = ["a", "b"]
    experiment_ids = _create_experiments(store, experiment_names)
    store.delete_experiment(experiment_ids[1])

    experiments = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == ["a", "Default"]
    experiments = store.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == ["b"]
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == ["b", "a", "Default"]


def test_search_experiments_filter_by_attribute(store: SqlAlchemyStore):
    experiment_names = ["a", "ab", "Abc"]
    _create_experiments(store, experiment_names)

    experiments = store.search_experiments(filter_string="name = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = store.search_experiments(filter_string="attribute.name = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = store.search_experiments(filter_string="attribute.`name` = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = store.search_experiments(filter_string="attribute.`name` != 'a'")
    assert [e.name for e in experiments] == ["Abc", "ab", "Default"]
    experiments = store.search_experiments(filter_string="name LIKE 'a%'")
    assert [e.name for e in experiments] == ["ab", "a"]
    experiments = store.search_experiments(filter_string="name ILIKE 'a%'")
    assert [e.name for e in experiments] == ["Abc", "ab", "a"]
    experiments = store.search_experiments(filter_string="name ILIKE 'a%' AND name ILIKE '%b'")
    assert [e.name for e in experiments] == ["ab"]


def test_search_experiments_filter_by_time_attribute(store: SqlAlchemyStore):
    # Sleep to ensure that the first experiment has a different creation_time than the default
    # experiment and eliminate flakiness.
    time.sleep(0.001)
    time_before_create1 = get_current_time_millis()
    exp_id1 = store.create_experiment("1")
    exp1 = store.get_experiment(exp_id1)
    time.sleep(0.001)
    time_before_create2 = get_current_time_millis()
    exp_id2 = store.create_experiment("2")
    exp2 = store.get_experiment(exp_id2)

    experiments = store.search_experiments(filter_string=f"creation_time = {exp1.creation_time}")
    assert [e.experiment_id for e in experiments] == [exp_id1]

    experiments = store.search_experiments(filter_string=f"creation_time != {exp1.creation_time}")
    assert [e.experiment_id for e in experiments] == [
        exp_id2,
        store.DEFAULT_EXPERIMENT_ID,
    ]

    experiments = store.search_experiments(filter_string=f"creation_time >= {time_before_create1}")
    assert [e.experiment_id for e in experiments] == [exp_id2, exp_id1]

    experiments = store.search_experiments(filter_string=f"creation_time < {time_before_create2}")
    assert [e.experiment_id for e in experiments] == [
        exp_id1,
        store.DEFAULT_EXPERIMENT_ID,
    ]

    # To avoid that the creation_time equals `now`, we wait one additional millisecond.
    time.sleep(0.001)
    now = get_current_time_millis()
    experiments = store.search_experiments(filter_string=f"creation_time >= {now}")
    assert experiments == []

    time.sleep(0.001)
    time_before_rename = get_current_time_millis()
    store.rename_experiment(exp_id1, "new_name")
    experiments = store.search_experiments(
        filter_string=f"last_update_time >= {time_before_rename}"
    )
    assert [e.experiment_id for e in experiments] == [exp_id1]

    experiments = store.search_experiments(
        filter_string=f"last_update_time <= {get_current_time_millis()}"
    )
    assert {e.experiment_id for e in experiments} == {
        exp_id1,
        exp_id2,
        store.DEFAULT_EXPERIMENT_ID,
    }

    experiments = store.search_experiments(
        filter_string=f"last_update_time = {exp2.last_update_time}"
    )
    assert [e.experiment_id for e in experiments] == [exp_id2]


def test_search_experiments_filter_by_tag(store: SqlAlchemyStore):
    experiments = [
        ("exp1", [ExperimentTag("key1", "value"), ExperimentTag("key2", "value")]),
        ("exp2", [ExperimentTag("key1", "vaLue"), ExperimentTag("key2", "vaLue")]),
        ("exp3", [ExperimentTag("k e y 1", "value")]),
    ]
    for name, tags in experiments:
        time.sleep(0.001)
        store.create_experiment(name, tags=tags)

    experiments = store.search_experiments(filter_string="tag.key1 = 'value'")
    assert [e.name for e in experiments] == ["exp1"]
    experiments = store.search_experiments(filter_string="tag.`k e y 1` = 'value'")
    assert [e.name for e in experiments] == ["exp3"]
    experiments = store.search_experiments(filter_string="tag.\"k e y 1\" = 'value'")
    assert [e.name for e in experiments] == ["exp3"]
    experiments = store.search_experiments(filter_string="tag.key1 != 'value'")
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.key1 != 'VALUE'")
    assert [e.name for e in experiments] == ["exp2", "exp1"]
    experiments = store.search_experiments(filter_string="tag.key1 LIKE 'val%'")
    assert [e.name for e in experiments] == ["exp1"]
    experiments = store.search_experiments(filter_string="tag.key1 LIKE '%Lue'")
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.key1 ILIKE '%alu%'")
    assert [e.name for e in experiments] == ["exp2", "exp1"]
    experiments = store.search_experiments(
        filter_string="tag.key1 LIKE 'va%' AND tag.key2 LIKE '%Lue'"
    )
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.KEY = 'value'")
    assert len(experiments) == 0


def test_search_experiments_filter_by_tag_is_null(store: SqlAlchemyStore):
    experiments = [
        ("exp1", [ExperimentTag("key1", "value"), ExperimentTag("key2", "value")]),
        ("exp2", [ExperimentTag("key1", "value")]),
        ("exp3", []),
    ]
    for name, tags in experiments:
        time.sleep(0.001)
        store.create_experiment(name, tags=tags)

    # IS NOT NULL: experiments that have key1
    results = store.search_experiments(filter_string="tag.key1 IS NOT NULL")
    assert [e.name for e in results] == ["exp2", "exp1"]

    # IS NULL: experiments that don't have key2 (includes Default)
    results = store.search_experiments(filter_string="tag.key2 IS NULL")
    assert [e.name for e in results] == ["exp3", "exp2", "Default"]

    # Combined IS NOT NULL and IS NULL
    results = store.search_experiments(filter_string="tag.key1 IS NOT NULL AND tag.key2 IS NULL")
    assert [e.name for e in results] == ["exp2"]

    # Combined with value filter
    results = store.search_experiments(filter_string="tag.key1 = 'value' AND tag.key2 IS NULL")
    assert [e.name for e in results] == ["exp2"]

    # Error: IS NULL on attribute
    with pytest.raises(MlflowException, match="IS NULL / IS NOT NULL is only supported for tags"):
        store.search_experiments(filter_string="name IS NULL")


def test_search_experiments_filter_by_attribute_and_tag(store: SqlAlchemyStore):
    store.create_experiment("exp1", tags=[ExperimentTag("a", "1"), ExperimentTag("b", "2")])
    store.create_experiment("exp2", tags=[ExperimentTag("a", "3"), ExperimentTag("b", "4")])
    experiments = store.search_experiments(filter_string="name ILIKE 'exp%' AND tags.a = '1'")
    assert [e.name for e in experiments] == ["exp1"]


def test_search_experiments_order_by(store: SqlAlchemyStore):
    experiment_names = ["x", "y", "z"]
    _create_experiments(store, experiment_names)

    experiments = store.search_experiments(order_by=["name"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

    experiments = store.search_experiments(order_by=["name ASC"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z"]

    experiments = store.search_experiments(order_by=["name DESC"])
    assert [e.name for e in experiments] == ["z", "y", "x", "Default"]

    experiments = store.search_experiments(order_by=["experiment_id DESC"])
    assert [e.name for e in experiments] == ["z", "y", "x", "Default"]

    experiments = store.search_experiments(order_by=["name", "experiment_id"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z"]


def test_search_experiments_order_by_time_attribute(store: SqlAlchemyStore):
    # Sleep to ensure that the first experiment has a different creation_time than the default
    # experiment and eliminate flakiness.
    time.sleep(0.001)
    exp_id1 = store.create_experiment("1")
    time.sleep(0.001)
    exp_id2 = store.create_experiment("2")

    experiments = store.search_experiments(order_by=["creation_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id1,
        exp_id2,
    ]

    experiments = store.search_experiments(order_by=["creation_time DESC"])
    assert [e.experiment_id for e in experiments] == [
        exp_id2,
        exp_id1,
        store.DEFAULT_EXPERIMENT_ID,
    ]

    experiments = store.search_experiments(order_by=["last_update_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id1,
        exp_id2,
    ]

    store.rename_experiment(exp_id1, "new_name")
    experiments = store.search_experiments(order_by=["last_update_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id2,
        exp_id1,
    ]


def test_search_experiments_max_results(store: SqlAlchemyStore):
    experiment_names = list(map(str, range(9)))
    _create_experiments(store, experiment_names)
    reversed_experiment_names = experiment_names[::-1]

    experiments = store.search_experiments()
    assert [e.name for e in experiments] == reversed_experiment_names + ["Default"]
    experiments = store.search_experiments(max_results=3)
    assert [e.name for e in experiments] == reversed_experiment_names[:3]


def test_search_experiments_max_results_validation(store: SqlAlchemyStore):
    with pytest.raises(
        MlflowException,
        match=r"Invalid value None for parameter 'max_results' supplied. "
        r"It must be a positive integer",
    ):
        store.search_experiments(max_results=None)
    with pytest.raises(
        MlflowException,
        match=r"Invalid value 0 for parameter 'max_results' supplied. "
        r"It must be a positive integer",
    ):
        store.search_experiments(max_results=0)
    with pytest.raises(
        MlflowException,
        match=r"Invalid value 1000000 for parameter 'max_results' supplied. "
        r"It must be at most 50000",
    ):
        store.search_experiments(max_results=1_000_000)


def test_search_experiments_pagination(store: SqlAlchemyStore):
    experiment_names = list(map(str, range(9)))
    _create_experiments(store, experiment_names)
    reversed_experiment_names = experiment_names[::-1]

    experiments = store.search_experiments(max_results=4)
    assert [e.name for e in experiments] == reversed_experiment_names[:4]
    assert experiments.token is not None

    experiments = store.search_experiments(max_results=4, page_token=experiments.token)
    assert [e.name for e in experiments] == reversed_experiment_names[4:8]
    assert experiments.token is not None

    experiments = store.search_experiments(max_results=4, page_token=experiments.token)
    assert [e.name for e in experiments] == reversed_experiment_names[8:] + ["Default"]
    assert experiments.token is None


def test_create_experiments(store: SqlAlchemyStore):
    with store.ManagedSessionMaker() as session:
        result = session.query(models.SqlExperiment).all()
        assert len(result) == 1
    time_before_create = get_current_time_millis()
    experiment_id = store.create_experiment(name="test exp")
    assert experiment_id == "1"
    with store.ManagedSessionMaker() as session:
        result = session.query(models.SqlExperiment).all()
        assert len(result) == 2

        test_exp = session.query(models.SqlExperiment).filter_by(name="test exp").first()
        assert str(test_exp.experiment_id) == experiment_id
        assert test_exp.name == "test exp"

    actual = store.get_experiment(experiment_id)
    assert actual.experiment_id == experiment_id
    assert actual.name == "test exp"
    assert actual.creation_time >= time_before_create
    assert actual.last_update_time == actual.creation_time

    with pytest.raises(MlflowException, match=r"'name' exceeds the maximum length"):
        store.create_experiment(name="x" * (MAX_EXPERIMENT_NAME_LENGTH + 1))


def test_create_experiment_with_tags_works_correctly(store: SqlAlchemyStore, workspaces_enabled):
    artifact_location = None if workspaces_enabled else "some location"
    experiment_id = store.create_experiment(
        name="test exp",
        artifact_location=artifact_location,
        tags=[ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
    )
    experiment = store.get_experiment(experiment_id)
    assert len(experiment.tags) == 2
    assert experiment.tags["key1"] == "val1"
    assert experiment.tags["key2"] == "val2"


def test_set_experiment_tag(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "setExperimentTagExp")
    tag = entities.ExperimentTag("tag0", "value0")
    new_tag = entities.RunTag("tag0", "value00000")
    store.set_experiment_tag(exp_id, tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value0"
    # test that updating a tag works
    store.set_experiment_tag(exp_id, new_tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value00000"
    # test that setting a tag on 1 experiment does not impact another experiment.
    exp_id_2 = _create_experiments(store, "setExperimentTagExp2")
    experiment2 = store.get_experiment(exp_id_2)
    assert len(experiment2.tags) == 0
    # setting a tag on different experiments maintains different values across experiments
    different_tag = entities.RunTag("tag0", "differentValue")
    store.set_experiment_tag(exp_id_2, different_tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value00000"
    experiment2 = store.get_experiment(exp_id_2)
    assert experiment2.tags["tag0"] == "differentValue"
    # test can set multi-line tags
    multi_line_Tag = entities.ExperimentTag("multiline tag", "value2\nvalue2\nvalue2")
    store.set_experiment_tag(exp_id, multi_line_Tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2"
    # test cannot set tags that are too long
    long_tag = entities.ExperimentTag("longTagKey", "a" * 100_001)
    with pytest.raises(MlflowException, match="exceeds the maximum length of 5000"):
        store.set_experiment_tag(exp_id, long_tag)
    # test can set tags that are somewhat long
    long_tag = entities.ExperimentTag("longTagKey", "a" * 4999)
    store.set_experiment_tag(exp_id, long_tag)
    # test cannot set tags on deleted experiments
    store.delete_experiment(exp_id)
    with pytest.raises(MlflowException, match="must be in the 'active' state"):
        store.set_experiment_tag(exp_id, entities.ExperimentTag("should", "notset"))


def test_delete_experiment_tag(store: SqlAlchemyStore):
    exp_id = _create_experiments(store, "setExperimentTagExp")
    tag = entities.ExperimentTag("tag0", "value0")
    store.set_experiment_tag(exp_id, tag)
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["tag0"] == "value0"
    # test that deleting a tag works
    store.delete_experiment_tag(exp_id, tag.key)
    experiment = store.get_experiment(exp_id)
    assert "tag0" not in experiment.tags


def test_rename_experiment(store: SqlAlchemyStore):
    new_name = "new name"
    experiment_id = _create_experiments(store, "test name")
    experiment = store.get_experiment(experiment_id)
    time.sleep(0.01)
    store.rename_experiment(experiment_id, new_name)

    renamed_experiment = store.get_experiment(experiment_id)

    assert renamed_experiment.name == new_name
    assert renamed_experiment.last_update_time > experiment.last_update_time

    with pytest.raises(MlflowException, match=r"'name' exceeds the maximum length"):
        store.rename_experiment(experiment_id, "x" * (MAX_EXPERIMENT_NAME_LENGTH + 1))


def test_restore_experiment(store: SqlAlchemyStore):
    experiment_id = _create_experiments(store, "helloexp")
    exp = store.get_experiment(experiment_id)
    assert exp.lifecycle_stage == entities.LifecycleStage.ACTIVE

    experiment_id = exp.experiment_id
    store.delete_experiment(experiment_id)

    deleted = store.get_experiment(experiment_id)
    assert deleted.experiment_id == experiment_id
    assert deleted.lifecycle_stage == entities.LifecycleStage.DELETED

    time.sleep(0.01)
    store.restore_experiment(exp.experiment_id)
    restored = store.get_experiment(experiment_id)
    assert restored.experiment_id == experiment_id
    assert restored.lifecycle_stage == entities.LifecycleStage.ACTIVE
    assert restored.last_update_time > deleted.last_update_time


def _assert_create_experiment_appends_to_artifact_uri_path_correctly(
    artifact_root_uri, expected_artifact_uri_format
):
    with TempDir() as tmp:
        dbfile_path = tmp.path("db")
        store = SqlAlchemyStore(
            db_uri="sqlite:///" + dbfile_path,
            default_artifact_root=artifact_root_uri,
        )
        exp_id = store.create_experiment(name="exp")
        exp = store.get_experiment(exp_id)

        store._dispose_engine()

        cwd = Path.cwd().as_posix()
        drive = Path.cwd().drive
        if is_windows() and expected_artifact_uri_format.startswith("file:"):
            cwd = f"/{cwd}"
            drive = f"{drive}/"
        assert exp.artifact_location == expected_artifact_uri_format.format(
            e=exp_id, cwd=cwd, drive=drive
        )


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        (
            "\\my_server/my_path/my_sub_path",
            "file:///{drive}my_server/my_path/my_sub_path/{e}",
        ),
        ("path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}"),
        ("/path/to/local/folder", "file:///{drive}path/to/local/folder/{e}"),
        ("#path/to/local/folder?", "file://{cwd}/{e}#path/to/local/folder?"),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}"),
        ("file:///path/to/local/folder", "file:///{drive}path/to/local/folder/{e}"),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}?param=value",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///{drive}path/to/local/folder/{e}?param=value#fragment",
        ),
    ],
)
def test_create_experiment_appends_to_artifact_local_path_file_uri_correctly_on_windows(
    input_uri, expected_uri
):
    _assert_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("path/to/local/folder", "{cwd}/path/to/local/folder/{e}"),
        ("/path/to/local/folder", "/path/to/local/folder/{e}"),
        ("#path/to/local/folder?", "{cwd}/#path/to/local/folder?/{e}"),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}"),
        ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}"),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}?param=value",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///path/to/local/folder/{e}?param=value#fragment",
        ),
    ],
)
def test_create_experiment_appends_to_artifact_local_path_file_uri_correctly(
    input_uri, expected_uri
):
    _assert_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("s3://bucket/path/to/root", "s3://bucket/path/to/root/{e}"),
        (
            "s3://bucket/path/to/root?creds=mycreds",
            "s3://bucket/path/to/root/{e}?creds=mycreds",
        ),
        (
            "dbscheme+driver://root@host/dbname?creds=mycreds#myfragment",
            "dbscheme+driver://root@host/dbname/{e}?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/{e}?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com/mydb?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/mydb/{e}?creds=mycreds#myfragment",
        ),
    ],
)
def test_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri):
    _assert_create_experiment_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)
