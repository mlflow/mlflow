import json
import os
import posixpath
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import List
from unittest import mock

import pytest

from mlflow.entities import (
    Dataset,
    DatasetInput,
    ExperimentTag,
    InputTag,
    LifecycleStage,
    Metric,
    Param,
    RunData,
    RunStatus,
    RunTag,
    ViewType,
    _DatasetSummary,
)
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils import insecure_hash
from mlflow.utils.file_utils import TempDir, path_to_local_file_uri, read_yaml, write_yaml
from mlflow.utils.mlflow_tags import MLFLOW_DATASET_CONTEXT, MLFLOW_LOGGED_MODELS, MLFLOW_RUN_NAME
from mlflow.utils.name_utils import _EXPERIMENT_ID_FIXED_WIDTH, _GENERATOR_PREDICATES
from mlflow.utils.os import is_windows
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import append_to_uri_path

from tests.helper_functions import random_int, random_str, safe_edit_yaml

FILESTORE_PACKAGE = "mlflow.store.tracking.file_store"


@pytest.fixture
def store(tmp_path):
    return FileStore(str(tmp_path.joinpath("mlruns")))


def create_experiments(store, experiment_names):
    ids = []
    for name in experiment_names:
        # ensure that the field `creation_time` is distinct for search ordering
        time.sleep(0.001)
        ids.append(store.create_experiment(name))
    return ids


def test_valid_root(store):
    store._check_root_dir()
    shutil.rmtree(store.root_directory)
    with pytest.raises(Exception, match=r"does not exist"):
        store._check_root_dir()


def test_attempting_to_remove_default_experiment(store):
    def _is_default_in_experiments(view_type):
        search_result = store.search_experiments(view_type=view_type)
        ids = [experiment.experiment_id for experiment in search_result]
        return FileStore.DEFAULT_EXPERIMENT_ID in ids

    assert _is_default_in_experiments(ViewType.ACTIVE_ONLY)

    # Ensure experiment deletion of default id raises
    with pytest.raises(MlflowException, match="Cannot delete the default experiment"):
        store.delete_experiment(FileStore.DEFAULT_EXPERIMENT_ID)


def test_search_experiments_view_type(store):
    experiment_names = ["a", "b"]
    experiment_ids = create_experiments(store, experiment_names)
    store.delete_experiment(experiment_ids[1])

    experiments = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == ["a", "Default"]
    experiments = store.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == ["b"]
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == ["b", "a", "Default"]


def test_search_experiments_filter_by_attribute(store):
    experiment_names = ["a", "ab", "Abc"]
    create_experiments(store, experiment_names)

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
    experiments = store.search_experiments(
        filter_string="name ILIKE 'a%'", order_by=["last_update_time asc"]
    )
    assert [e.name for e in experiments] == ["a", "ab", "Abc"]
    experiments = store.search_experiments(filter_string="name ILIKE 'a%'")
    assert [e.name for e in experiments] == ["Abc", "ab", "a"]
    experiments = store.search_experiments(filter_string="name ILIKE 'a%' AND name ILIKE '%b'")
    assert [e.name for e in experiments] == ["ab"]


def test_search_experiments_filter_by_time_attribute(store):
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
    assert [e.experiment_id for e in experiments] == [exp_id2, store.DEFAULT_EXPERIMENT_ID]

    experiments = store.search_experiments(filter_string=f"creation_time >= {time_before_create1}")
    assert [e.experiment_id for e in experiments] == [exp_id2, exp_id1]

    experiments = store.search_experiments(filter_string=f"creation_time < {time_before_create2}")
    assert [e.experiment_id for e in experiments] == [exp_id1, store.DEFAULT_EXPERIMENT_ID]

    now = get_current_time_millis()
    experiments = store.search_experiments(filter_string=f"creation_time > {now}")
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


def test_search_experiments_filter_by_attribute_and_tag(store):
    store.create_experiment("exp1", tags=[ExperimentTag("a", "1"), ExperimentTag("b", "2")])
    store.create_experiment("exp2", tags=[ExperimentTag("a", "3"), ExperimentTag("b", "4")])
    experiments = store.search_experiments(filter_string="name ILIKE 'exp%' AND tag.a = '1'")
    assert [e.name for e in experiments] == ["exp1"]


def test_search_experiments_filter_by_tag(store):
    experiments = [
        ("exp1", [ExperimentTag("key", "value")]),
        ("exp2", [ExperimentTag("key", "vaLue")]),
        ("exp3", [ExperimentTag("k e y", "value")]),
    ]
    for name, tags in experiments:
        # sleep to enforce deterministic ordering based on last_update_time (creation_time due to
        # no mutation of experiment state)
        time.sleep(0.001)
        store.create_experiment(name, tags=tags)

    experiments = store.search_experiments(filter_string="tag.key = 'value'")
    assert [e.name for e in experiments] == ["exp1"]
    experiments = store.search_experiments(filter_string="tag.`k e y` = 'value'")
    assert [e.name for e in experiments] == ["exp3"]
    experiments = store.search_experiments(filter_string="tag.\"k e y\" = 'value'")
    assert [e.name for e in experiments] == ["exp3"]
    experiments = store.search_experiments(filter_string="tag.key != 'value'")
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.key LIKE 'val%'")
    assert [e.name for e in experiments] == ["exp1"]
    experiments = store.search_experiments(filter_string="tag.key LIKE '%Lue'")
    assert [e.name for e in experiments] == ["exp2"]
    experiments = store.search_experiments(filter_string="tag.key ILIKE '%alu%'")
    assert [e.name for e in experiments] == ["exp2", "exp1"]
    experiments = store.search_experiments(
        filter_string="tag.key LIKE 'va%' AND tags.key LIKE '%Lue'"
    )
    assert [e.name for e in experiments] == ["exp2"]


def test_search_experiments_order_by(store):
    experiment_names = ["x", "y", "z"]
    create_experiments(store, experiment_names)

    # Test the case where an experiment does not have a creation time by simulating a time of
    # `None`. This is applicable to experiments created in older versions of MLflow where the
    # `creation_time` attribute did not exist
    with mock.patch(
        "mlflow.store.tracking.file_store.get_current_time_millis",
        return_value=None,
    ):
        store.create_experiment("n")

    experiments = store.search_experiments(order_by=["name"])
    assert [e.name for e in experiments] == ["Default", "n", "x", "y", "z"]

    experiments = store.search_experiments(order_by=["name ASC"])
    assert [e.name for e in experiments] == ["Default", "n", "x", "y", "z"]

    experiments = store.search_experiments(order_by=["name DESC"])
    assert [e.name for e in experiments] == ["z", "y", "x", "n", "Default"]

    experiments = store.search_experiments(order_by=["creation_time DESC"])
    assert [e.name for e in experiments] == ["z", "y", "x", "Default", "n"]

    experiments = store.search_experiments(order_by=["creation_time ASC"])
    assert [e.name for e in experiments] == ["Default", "x", "y", "z", "n"]

    experiments = store.search_experiments(order_by=["name", "last_update_time asc"])
    assert [e.name for e in experiments] == ["Default", "n", "x", "y", "z"]


def test_search_experiments_order_by_time_attribute(store):
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

    time.sleep(0.001)
    store.rename_experiment(exp_id1, "new_name")
    experiments = store.search_experiments(order_by=["last_update_time"])
    assert [e.experiment_id for e in experiments] == [
        store.DEFAULT_EXPERIMENT_ID,
        exp_id2,
        exp_id1,
    ]


def test_search_experiments_max_results(store):
    experiment_names = list(map(str, range(9)))
    create_experiments(store, experiment_names)
    reversed_experiment_names = experiment_names[::-1]

    experiments = store.search_experiments()
    assert [e.name for e in experiments] == reversed_experiment_names + ["Default"]
    experiments = store.search_experiments(max_results=3)
    assert [e.name for e in experiments] == reversed_experiment_names[:3]


def test_search_experiments_max_results_validation(store):
    with pytest.raises(MlflowException, match=r"It must be a positive integer, but got None"):
        store.search_experiments(max_results=None)
    with pytest.raises(MlflowException, match=r"It must be a positive integer, but got 0"):
        store.search_experiments(max_results=0)
    with pytest.raises(MlflowException, match=r"It must be at most \d+, but got 1000000"):
        store.search_experiments(max_results=1_000_000)


def test_search_experiments_pagination(store):
    experiment_names = list(map(str, range(9)))
    create_experiments(store, experiment_names)
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


def _verify_experiment(fs, exp_id, exp_data):
    exp = fs.get_experiment(exp_id)
    assert exp.experiment_id == exp_id
    assert exp.name == exp_data[exp_id]["name"]
    assert exp.artifact_location == exp_data[exp_id]["artifact_location"]


def _verify_logged(store, run_id, metrics, params, tags):
    run = store.get_run(run_id)
    all_metrics = sum([store.get_metric_history(run_id, key) for key in run.data.metrics], [])
    assert len(all_metrics) == len(metrics)
    logged_metrics = [(m.key, m.value, m.timestamp, m.step) for m in all_metrics]
    assert set(logged_metrics) == {(m.key, m.value, m.timestamp, m.step) for m in metrics}
    logged_tags = set(run.data.tags.items())
    assert {(tag.key, tag.value) for tag in tags} <= logged_tags
    assert len(run.data.params) == len(params)
    assert set(run.data.params.items()) == {(param.key, param.value) for param in params}


def _create_root(store):
    test_root = store.root_directory
    experiments = [str(random_int(100, int(1e9))) for _ in range(3)]
    exp_data = {}
    run_data = {}
    # Include default experiment
    experiments.append(FileStore.DEFAULT_EXPERIMENT_ID)
    default_exp_folder = os.path.join(test_root, str(FileStore.DEFAULT_EXPERIMENT_ID))
    if os.path.exists(default_exp_folder):
        shutil.rmtree(default_exp_folder)

    for exp in experiments:
        # create experiment
        exp_folder = os.path.join(test_root, str(exp))
        os.makedirs(exp_folder)
        current_time = get_current_time_millis()
        d = {
            "experiment_id": exp,
            "name": random_str(),
            "artifact_location": exp_folder,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "creation_time": current_time,
            "last_update_time": current_time,
        }
        exp_data[exp] = d
        write_yaml(exp_folder, FileStore.META_DATA_FILE_NAME, d)
        # add runs
        exp_data[exp]["runs"] = []
        for _ in range(2):
            run_id = uuid.uuid4().hex
            exp_data[exp]["runs"].append(run_id)
            run_folder = os.path.join(exp_folder, run_id)
            os.makedirs(run_folder)
            run_info = {
                "run_uuid": run_id,
                "run_id": run_id,
                "run_name": "name",
                "experiment_id": exp,
                "user_id": random_str(random_int(10, 25)),
                "status": random.choice(RunStatus.all_status()),
                "start_time": random_int(1, 10),
                "end_time": random_int(20, 30),
                "deleted_time": random_int(20, 30),
                "tags": [],
                "artifact_uri": os.path.join(run_folder, FileStore.ARTIFACTS_FOLDER_NAME),
                "lifecycle_stage": LifecycleStage.ACTIVE,
            }
            write_yaml(run_folder, FileStore.META_DATA_FILE_NAME, run_info)
            run_data[run_id] = run_info
            # tags
            os.makedirs(os.path.join(run_folder, FileStore.TAGS_FOLDER_NAME))
            # params
            params_folder = os.path.join(run_folder, FileStore.PARAMS_FOLDER_NAME)
            os.makedirs(params_folder)
            params = {}
            for _ in range(5):
                param_name = random_str(random_int(10, 12))
                param_value = random_str(random_int(10, 15))
                param_file = os.path.join(params_folder, param_name)
                with open(param_file, "w") as f:
                    f.write(param_value)
                params[param_name] = param_value
            run_data[run_id]["params"] = params
            # metrics
            metrics_folder = os.path.join(run_folder, FileStore.METRICS_FOLDER_NAME)
            os.makedirs(metrics_folder)
            metrics = {}
            for _ in range(3):
                metric_name = random_str(random_int(10, 12))
                timestamp = get_current_time_millis()
                metric_file = os.path.join(metrics_folder, metric_name)
                values = []
                for _ in range(10):
                    metric_value = random_int(100, 2000)
                    timestamp += random_int(10000, 2000000)
                    values.append((timestamp, metric_value))
                    with open(metric_file, "a") as f:
                        f.write(f"{timestamp} {metric_value}\n")
                metrics[metric_name] = values
            run_data[run_id]["metrics"] = metrics
            # artifacts
            os.makedirs(os.path.join(run_folder, FileStore.ARTIFACTS_FOLDER_NAME))

    return experiments, exp_data, run_data


def create_test_run(store):
    return store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )


def test_record_logged_model(store):
    run_id = create_test_run(store).info.run_id
    m = Model(artifact_path="model/path", run_id=run_id, flavors={"tf": "flavor body"})
    store.record_logged_model(run_id, m)
    _verify_logged(
        store,
        run_id=run_id,
        params=[],
        metrics=[],
        tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict()]))],
    )
    m2 = Model(artifact_path="some/other/path", run_id=run_id, flavors={"R": {"property": "value"}})
    store.record_logged_model(run_id, m2)
    _verify_logged(
        store,
        run_id,
        params=[],
        metrics=[],
        tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict(), m2.to_dict()]))],
    )
    m3 = Model(
        artifact_path="some/other/path2", run_id=run_id, flavors={"R2": {"property": "value"}}
    )
    store.record_logged_model(run_id, m3)
    _verify_logged(
        store,
        run_id,
        params=[],
        metrics=[],
        tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict(), m2.to_dict(), m3.to_dict()]))],
    )
    with pytest.raises(
        TypeError,
        match="Argument 'mlflow_model' should be mlflow.models.Model, got '<class 'dict'>'",
    ):
        store.record_logged_model(run_id, m.to_dict())


def test_get_experiment(store):
    experiments, exp_data, _ = _create_root(store)
    for exp_id in experiments:
        _verify_experiment(store, exp_id, exp_data)

    # test that fake experiments dont exist.
    # look for random experiment ids between 8000, 15000 since created ones are (100, 2000)
    for exp_id in {random_int(8000, 15000) for x in range(20)}:
        with pytest.raises(Exception, match=f"Could not find experiment with ID {exp_id}"):
            store.get_experiment(str(exp_id))


def test_get_experiment_int_experiment_id_backcompat(store):
    _, exp_data, _ = _create_root(store)
    exp_id = FileStore.DEFAULT_EXPERIMENT_ID
    root_dir = os.path.join(store.root_directory, exp_id)
    with safe_edit_yaml(root_dir, "meta.yaml", _experiment_id_edit_func):
        _verify_experiment(store, exp_id, exp_data)


def test_get_experiment_retries_for_transient_empty_yaml_read(store):
    exp_name = random_str()
    exp_id = store.create_experiment(exp_name)

    mock_empty_call_count = 0

    def mock_read_yaml_impl(*args, **kwargs):
        nonlocal mock_empty_call_count
        if mock_empty_call_count < 2:
            mock_empty_call_count += 1
            return None
        else:
            return read_yaml(*args, **kwargs)

    with mock.patch(
        "mlflow.store.tracking.file_store.read_yaml", side_effect=mock_read_yaml_impl
    ) as mock_read_yaml:
        fetched_experiment = store.get_experiment(exp_id)
        assert fetched_experiment.experiment_id == exp_id
        assert fetched_experiment.name == exp_name
        assert mock_read_yaml.call_count == 3


def test_get_experiment_by_name(store):
    experiments, exp_data, _ = _create_root(store)
    for exp_id in experiments:
        name = exp_data[exp_id]["name"]
        exp = store.get_experiment_by_name(name)
        assert exp.experiment_id == exp_id
        assert exp.name == exp_data[exp_id]["name"]
        assert exp.artifact_location == exp_data[exp_id]["artifact_location"]

    # test that fake experiments dont exist.
    # look up experiments with names of length 15 since created ones are of length 10
    for exp_names in {random_str(15) for x in range(20)}:
        exp = store.get_experiment_by_name(exp_names)
        assert exp is None


def test_create_additional_experiment_generates_random_fixed_length_id(store):
    store._get_active_experiments = mock.Mock(return_value=[])
    store._get_deleted_experiments = mock.Mock(return_value=[])
    store._create_experiment_with_id = mock.Mock()
    store.create_experiment(random_str())
    store._create_experiment_with_id.assert_called_once()
    experiment_id = store._create_experiment_with_id.call_args[0][1]
    assert len(experiment_id) == _EXPERIMENT_ID_FIXED_WIDTH


def test_create_experiment(store):
    # fs = FileStore(helper.test_root)

    # Error cases
    with pytest.raises(Exception, match="Invalid experiment name: 'None'"):
        store.create_experiment(None)
    with pytest.raises(Exception, match="Invalid experiment name: ''"):
        store.create_experiment("")
    name = random_str(25)  # since existing experiments are 10 chars long
    time_before_create = get_current_time_millis()
    created_id = store.create_experiment(name)
    # test that newly created experiment id is random but of a fixed length
    assert len(created_id) == _EXPERIMENT_ID_FIXED_WIDTH

    # get the new experiment (by id) and verify (by name)
    exp1 = store.get_experiment(created_id)
    assert exp1.name == name
    assert exp1.artifact_location == path_to_local_file_uri(
        posixpath.join(store.root_directory, created_id)
    )
    assert exp1.creation_time >= time_before_create
    assert exp1.last_update_time == exp1.creation_time

    # get the new experiment (by name) and verify (by id)
    exp2 = store.get_experiment_by_name(name)
    assert exp2.experiment_id == created_id
    assert exp2.creation_time == exp1.creation_time
    assert exp2.last_update_time == exp1.last_update_time


def test_create_experiment_with_tags_works_correctly(store):
    created_id = store.create_experiment(
        "heresAnExperiment",
        "heresAnArtifact",
        [ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
    )
    experiment = store.get_experiment(created_id)
    assert len(experiment.tags) == 2
    assert experiment.tags["key1"] == "val1"
    assert experiment.tags["key2"] == "val2"


def test_create_duplicate_experiments(store):
    experiments, exp_data, _ = _create_root(store)
    for exp_id in experiments:
        name = exp_data[exp_id]["name"]
        with pytest.raises(Exception, match=f"Experiment '{name}' already exists"):
            store.create_experiment(name)


def _extract_ids(experiments):
    return [e.experiment_id for e in experiments]


def test_delete_restore_experiment(store):
    experiments, _, _ = _create_root(store)
    exp1_id = experiments[random_int(0, len(experiments) - 2)]  # never select default experiment
    exp1 = store.get_experiment(exp1_id)

    # test deleting experiment
    store.delete_experiment(exp1_id)
    assert exp1_id not in _extract_ids(store.search_experiments(view_type=ViewType.ACTIVE_ONLY))
    assert exp1_id in _extract_ids(store.search_experiments(view_type=ViewType.DELETED_ONLY))
    assert exp1_id in _extract_ids(store.search_experiments(view_type=ViewType.ALL))
    deleted_exp1 = store.get_experiment(exp1_id)
    assert deleted_exp1.last_update_time > exp1.last_update_time
    assert deleted_exp1.lifecycle_stage == LifecycleStage.DELETED

    # test if setting lifecycle_stage is persisted correctly
    deleted_exp1_dir = store._get_experiment_path(
        experiment_id=exp1_id, view_type=ViewType.DELETED_ONLY
    )
    deleted_exp1_meta = FileStore._read_yaml(
        root=deleted_exp1_dir, file_name=FileStore.META_DATA_FILE_NAME
    )
    assert deleted_exp1_meta["lifecycle_stage"] == LifecycleStage.DELETED
    for run in store.search_runs(
        experiment_ids=[exp1_id], filter_string="", run_view_type=ViewType.ALL
    ):
        assert run.info.lifecycle_stage == LifecycleStage.DELETED

    # test restoring experiment
    store.restore_experiment(exp1_id)
    assert exp1_id in _extract_ids(store.search_experiments(view_type=ViewType.ACTIVE_ONLY))
    assert exp1_id not in _extract_ids(store.search_experiments(view_type=ViewType.DELETED_ONLY))
    assert exp1_id in _extract_ids(store.search_experiments(view_type=ViewType.ALL))
    restored1_exp1 = store.get_experiment(exp1_id)
    assert restored1_exp1.experiment_id == exp1_id
    assert restored1_exp1.name == exp1.name
    assert restored1_exp1.last_update_time > exp1.last_update_time
    assert restored1_exp1.lifecycle_stage == LifecycleStage.ACTIVE
    restored2_exp1 = store.get_experiment_by_name(exp1.name)
    assert restored2_exp1.experiment_id == exp1_id
    assert restored2_exp1.name == exp1.name

    # test if setting lifecycle_stage is persisted correctly
    restored_exp1_dir = store._get_experiment_path(
        experiment_id=exp1_id, view_type=ViewType.ACTIVE_ONLY
    )
    restored_exp1_meta = FileStore._read_yaml(
        root=restored_exp1_dir, file_name=FileStore.META_DATA_FILE_NAME
    )
    assert restored_exp1_meta["lifecycle_stage"] == LifecycleStage.ACTIVE
    for run in store.search_runs(
        experiment_ids=[exp1_id], filter_string="", run_view_type=ViewType.ALL
    ):
        assert run.info.lifecycle_stage == LifecycleStage.ACTIVE


def test_rename_experiment(store):
    experiments, _, _ = _create_root(store)
    exp_id = store.create_experiment("test_rename")

    # Error cases
    with pytest.raises(Exception, match="Invalid experiment name: 'None'"):
        store.rename_experiment(exp_id, None)
    # test that names of existing experiments are checked before renaming
    other_exp_id = None
    for exp in experiments:
        if exp != exp_id:
            other_exp_id = exp
            break
    name = store.get_experiment(other_exp_id).name
    with pytest.raises(Exception, match=f"Experiment '{name}' already exists"):
        store.rename_experiment(exp_id, name)

    exp_name = store.get_experiment(exp_id).name
    new_name = exp_name + "!!!"
    assert exp_name != new_name
    assert store.get_experiment(exp_id).name == exp_name
    store.rename_experiment(exp_id, new_name)
    assert store.get_experiment(exp_id).name == new_name

    # Ensure that we cannot rename deleted experiments.
    store.delete_experiment(exp_id)
    with pytest.raises(
        Exception, match="Cannot rename experiment in non-active lifecycle stage"
    ) as e:
        store.rename_experiment(exp_id, exp_name)
    assert "non-active lifecycle" in str(e.value)
    assert store.get_experiment(exp_id).name == new_name

    # Restore the experiment, and confirm that we can now rename it.
    exp1 = store.get_experiment(exp_id)
    time.sleep(0.01)
    store.restore_experiment(exp_id)
    restored_exp1 = store.get_experiment(exp_id)
    assert restored_exp1.name == new_name
    assert restored_exp1.last_update_time > exp1.last_update_time

    exp1 = store.get_experiment(exp_id)
    time.sleep(0.01)
    store.rename_experiment(exp_id, exp_name)
    renamed_exp1 = store.get_experiment(exp_id)
    assert renamed_exp1.name == exp_name
    assert renamed_exp1.last_update_time > exp1.last_update_time


def test_delete_restore_run(store):
    experiments, exp_data, _ = _create_root(store)
    exp_id = experiments[random_int(0, len(experiments) - 1)]
    run_id = exp_data[exp_id]["runs"][0]
    _, run_dir = store._find_run_root(run_id)
    # Should not throw.
    assert store.get_run(run_id).info.lifecycle_stage == "active"
    # Verify that run deletion is idempotent by deleting twice
    store.delete_run(run_id)
    store.delete_run(run_id)
    assert store.get_run(run_id).info.lifecycle_stage == "deleted"
    meta = read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
    assert "deleted_time" in meta
    assert meta["deleted_time"] is not None
    # Verify that run restoration is idempotent by restoring twice
    store.restore_run(run_id)
    store.restore_run(run_id)
    assert store.get_run(run_id).info.lifecycle_stage == "active"
    meta = read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
    assert "deleted_time" not in meta


def test_hard_delete_run(store):
    # fs = FileStore(helper.test_root)
    experiments, exp_data, _ = _create_root(store)
    exp_id = experiments[random_int(0, len(experiments) - 1)]
    run_id = exp_data[exp_id]["runs"][0]
    store._hard_delete_run(run_id)
    with pytest.raises(MlflowException, match=f"Run '{run_id}' not found"):
        store.get_run(run_id)
    with pytest.raises(MlflowException, match=f"Run '{run_id}' not found"):
        store.get_all_tags(run_id)
    with pytest.raises(MlflowException, match=f"Run '{run_id}' not found"):
        store.get_all_metrics(run_id)
    with pytest.raises(MlflowException, match=f"Run '{run_id}' not found"):
        store.get_all_params(run_id)


def test_get_deleted_runs(store):
    experiments, exp_data, _ = _create_root(store)
    exp_id = experiments[0]
    run_id = exp_data[exp_id]["runs"][0]
    store.delete_run(run_id)
    deleted_runs = store._get_deleted_runs()
    assert len(deleted_runs) == 1
    assert deleted_runs[0] == run_id


def test_create_run_in_deleted_experiment(store):
    exp_id = store.create_experiment("test")
    store.delete_experiment(exp_id)
    with pytest.raises(Exception, match="Could not create run under non-active experiment"):
        store.create_run(exp_id, "user", 0, [], "name")


def test_create_run_returns_expected_run_data(store):
    no_tags_run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name=None,
    )
    assert isinstance(no_tags_run.data, RunData)
    assert len(no_tags_run.data.tags) == 1

    run_name = no_tags_run.info.run_name
    assert run_name.split("-")[0] in _GENERATOR_PREDICATES

    run_name = no_tags_run.info.run_name
    assert run_name.split("-")[0] in _GENERATOR_PREDICATES

    tags_dict = {
        "my_first_tag": "first",
        "my-second-tag": "2nd",
    }
    tags_entities = [RunTag(key, value) for key, value in tags_dict.items()]
    tags_run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=tags_entities,
        run_name=None,
    )
    assert isinstance(tags_run.data, RunData)
    assert tags_run.data.tags == {**tags_dict, MLFLOW_RUN_NAME: tags_run.info.run_name}

    name_empty_str_run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=tags_entities,
        run_name="",
    )
    run_name = name_empty_str_run.info.run_name
    assert run_name.split("-")[0] in _GENERATOR_PREDICATES


def test_create_run_sets_name(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="my name",
    )

    run = store.get_run(run.info.run_id)
    assert run.info.run_name == "my name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "my name"

    run_id = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        run_name=None,
        tags=[RunTag(MLFLOW_RUN_NAME, "test")],
    ).info.run_id
    run = store.get_run(run_id)
    assert run.info.run_name == "test"

    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
            "different values (run_name='my name', mlflow.runName='test')."
        ),
    ):
        store.create_run(
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
            user_id="user",
            start_time=0,
            run_name="my name",
            tags=[RunTag(MLFLOW_RUN_NAME, "test")],
        )


def _experiment_id_edit_func(old_dict):
    old_dict["experiment_id"] = int(old_dict["experiment_id"])
    return old_dict


def _verify_run(store, run_id, run_data):
    run = store.get_run(run_id)
    run_info = run_data[run_id]
    run_info.pop("metrics", None)
    run_info.pop("params", None)
    run_info.pop("tags", None)
    run_info.pop("deleted_time", None)
    run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
    run_info["status"] = RunStatus.to_string(run_info["status"])
    # get a copy of run_info as we need to remove the `deleted_time`
    # key without actually deleting it from self.run_data
    _run_info = run_info.copy()
    _run_info.pop("deleted_time", None)
    assert _run_info == dict(run.info)


def test_get_run(store):
    experiments, exp_data, run_data = _create_root(store)
    for exp_id in experiments:
        runs = exp_data[exp_id]["runs"]
        for run_id in runs:
            _verify_run(store, run_id, run_data)


def test_get_run_returns_name_in_info(store):
    run_id = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="my name",
    ).info.run_id

    get_run = store.get_run(run_id)
    assert get_run.info.run_name == "my name"


def test_get_run_retries_for_transient_empty_yaml_read(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )

    mock_empty_call_count = 0

    def mock_read_yaml_impl(*args, **kwargs):
        nonlocal mock_empty_call_count
        if mock_empty_call_count < 2:
            mock_empty_call_count += 1
            return None
        else:
            return read_yaml(*args, **kwargs)

    with mock.patch(
        "mlflow.store.tracking.file_store.read_yaml", side_effect=mock_read_yaml_impl
    ) as mock_read_yaml:
        fetched_run = store.get_run(run.info.run_id)
        assert fetched_run.info.run_id == run.info.run_id
        assert fetched_run.info.artifact_uri == run.info.artifact_uri
        assert mock_read_yaml.call_count == 3


def test_get_run_int_experiment_id_backcompat(store):
    _, exp_data, run_data = _create_root(store)
    exp_id = FileStore.DEFAULT_EXPERIMENT_ID
    run_id = exp_data[exp_id]["runs"][0]
    root_dir = os.path.join(store.root_directory, exp_id, run_id)
    with safe_edit_yaml(root_dir, "meta.yaml", _experiment_id_edit_func):
        _verify_run(store, run_id, run_data)


def test_update_run_renames_run(store):
    run_id = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="first name",
    ).info.run_id
    store.update_run_info(run_id, RunStatus.FINISHED, 1000, "new name")
    get_run = store.get_run(run_id)
    assert get_run.info.run_name == "new name"


def test_update_run_does_not_rename_run_with_none_name(store):
    run_id = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="first name",
    ).info.run_id
    store.update_run_info(run_id, RunStatus.FINISHED, 1000, None)
    get_run = store.get_run(run_id)
    assert get_run.info.run_name == "first name"


def test_log_metric_allows_multiple_values_at_same_step_and_run_data_uses_max_step_value(store):
    run_id = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="first name",
    ).info.run_id

    metric_name = "test-metric-1"
    # Check that we get the max of (step, timestamp, value) in that order
    tuples_to_log = [
        (0, 100, 1000),
        (3, 40, 100),  # larger step wins even though it has smaller value
        (3, 50, 10),  # larger timestamp wins even though it has smaller value
        (3, 50, 20),  # tiebreak by max value
        (3, 50, 20),  # duplicate metrics with same (step, timestamp, value) are ok
        # verify that we can log steps out of order / negative steps
        (-3, 900, 900),
        (-1, 800, 800),
    ]
    for step, timestamp, value in reversed(tuples_to_log):
        store.log_metric(run_id, Metric(metric_name, value, timestamp, step))

    metric_history = store.get_metric_history(run_id, metric_name)
    logged_tuples = [(m.step, m.timestamp, m.value) for m in metric_history]
    assert set(logged_tuples) == set(tuples_to_log)

    run_data = store.get_run(run_id).data
    run_metrics = run_data.metrics
    assert len(run_metrics) == 1
    assert run_metrics[metric_name] == 20
    metric_obj = run_data._metric_objs[0]
    assert metric_obj.key == metric_name
    assert metric_obj.step == 3
    assert metric_obj.timestamp == 50
    assert metric_obj.value == 20


def test_log_metric_with_non_numeric_value_raises_exception(store):
    run_id = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="first name",
    ).info.run_id
    with pytest.raises(MlflowException, match=r"Got invalid value string for metric"):
        store.log_metric(run_id, Metric("test", "string", 0, 0))


def test_get_all_metrics(store):
    experiments, exp_data, run_data = _create_root(store)
    for exp_id in experiments:
        runs = exp_data[exp_id]["runs"]
        for run_id in runs:
            run_info = run_data[run_id]
            metrics = store.get_all_metrics(run_id)
            metrics_dict = run_info.pop("metrics")
            for metric in metrics:
                expected_timestamp, expected_value = max(metrics_dict[metric.key])
                assert metric.timestamp == expected_timestamp
                assert metric.value == expected_value


def test_get_metric_history(store):
    experiments, exp_data, run_data = _create_root(store)
    for exp_id in experiments:
        runs = exp_data[exp_id]["runs"]
        for run_id in runs:
            run_info = run_data[run_id]
            metrics = run_info.pop("metrics")
            for metric_name, values in metrics.items():
                metric_history = store.get_metric_history(run_id, metric_name)
                sorted_values = sorted(values, reverse=True)
                for metric in metric_history:
                    timestamp, metric_value = sorted_values.pop()
                    assert metric.timestamp == timestamp
                    assert metric.key == metric_name
                    assert metric.value == metric_value


def test_get_metric_history_paginated_request_raises(store):
    with pytest.raises(
        MlflowException,
        match="The FileStore backend does not support pagination for the `get_metric_history` "
        "API.",
    ):
        store.get_metric_history("fake_run", "fake_metric", max_results=50, page_token="42")


def _search(
    fs,
    experiment_id,
    filter_str=None,
    run_view_type=ViewType.ALL,
    max_results=SEARCH_MAX_RESULTS_DEFAULT,
):
    return [
        r.info.run_id
        for r in fs.search_runs([experiment_id], filter_str, run_view_type, max_results)
    ]


def test_search_runs(store):
    # replace with test with code is implemented
    experiments, _, _ = _create_root(store)
    # Expect 2 runs for each experiment
    assert len(_search(store, experiments[0], run_view_type=ViewType.ACTIVE_ONLY)) == 2
    assert len(_search(store, experiments[0])) == 2
    assert len(_search(store, experiments[0], run_view_type=ViewType.DELETED_ONLY)) == 0


def test_search_tags(store):
    experiments, _, _ = _create_root(store)
    experiment_id = experiments[0]
    r1 = store.create_run(experiment_id, "user", 0, [], "name").info.run_id
    r2 = store.create_run(experiment_id, "user", 0, [], "name").info.run_id

    store.set_tag(r1, RunTag("generic_tag", "p_val"))
    store.set_tag(r2, RunTag("generic_tag", "p_val"))

    store.set_tag(r1, RunTag("generic_2", "some value"))
    store.set_tag(r2, RunTag("generic_2", "another value"))

    store.set_tag(r1, RunTag("p_a", "abc"))
    store.set_tag(r2, RunTag("p_b", "ABC"))

    # test search returns both runs
    assert sorted(
        [r1, r2],
    ) == sorted(_search(store, experiment_id, filter_str="tags.generic_tag = 'p_val'"))
    # test search returns appropriate run (same key different values per run)
    assert _search(store, experiment_id, filter_str="tags.generic_2 = 'some value'") == [r1]
    assert _search(store, experiment_id, filter_str="tags.generic_2='another value'") == [r2]
    assert _search(store, experiment_id, filter_str="tags.generic_tag = 'wrong_val'") == []
    assert _search(store, experiment_id, filter_str="tags.generic_tag != 'p_val'") == []
    assert sorted([r1, r2]) == sorted(
        _search(store, experiment_id, filter_str="tags.generic_tag != 'wrong_val'"),
    )
    assert sorted([r1, r2]) == sorted(
        _search(store, experiment_id, filter_str="tags.generic_2 != 'wrong_val'"),
    )
    assert _search(store, experiment_id, filter_str="tags.p_a = 'abc'") == [r1]
    assert _search(store, experiment_id, filter_str="tags.p_b = 'ABC'") == [r2]

    assert _search(store, experiment_id, filter_str="tags.generic_2 LIKE '%other%'") == [r2]
    assert _search(store, experiment_id, filter_str="tags.generic_2 LIKE 'other%'") == []
    assert _search(store, experiment_id, filter_str="tags.generic_2 LIKE '%other'") == []
    assert _search(store, experiment_id, filter_str="tags.generic_2 ILIKE '%OTHER%'") == [r2]


def test_search_with_max_results(store):
    exp = store.create_experiment("search_with_max_results")

    runs = [store.create_run(exp, "user", r, [], "name").info.run_id for r in range(10)]
    runs.reverse()

    assert runs[:10] == _search(store, exp)
    for n in [0, 1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
        assert runs[: min(1200, n)] == _search(store, exp, max_results=n)

    with pytest.raises(
        MlflowException, match="Invalid value for request parameter max_results. It "
    ):
        _search(store, exp, None, max_results=int(1e10))


def test_search_with_deterministic_max_results(store):
    exp = store.create_experiment("test_search_with_deterministic_max_results")

    # Create 10 runs with the same start_time.
    # Sort based on run_id
    runs = sorted([store.create_run(exp, "user", 1000, [], "name").info.run_id for r in range(10)])
    for n in [0, 1, 2, 4, 8, 10, 20]:
        assert runs[: min(10, n)] == _search(store, exp, max_results=n)


def test_search_runs_pagination(store):
    exp = store.create_experiment("test_search_runs_pagination")
    # test returned token behavior
    runs = sorted([store.create_run(exp, "user", 1000, [], "name").info.run_id for r in range(10)])
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4)
    assert [r.info.run_id for r in result] == runs[0:4]
    assert result.token is not None
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[4:8]
    assert result.token is not None
    result = store.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[8:]
    assert result.token is None


def test_search_runs_run_name(store):
    exp_id = store.create_experiment("test_search_runs_pagination")
    run1 = store.create_run(exp_id, user_id="user", start_time=1000, tags=[], run_name="run_name1")
    run2 = store.create_run(exp_id, user_id="user", start_time=1000, tags=[], run_name="run_name2")
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]
    result = store.search_runs(
        [exp_id],
        filter_string="tags.`mlflow.runName` = 'run_name2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run2.info.run_id]

    store.update_run_info(
        run1.info.run_id,
        RunStatus.FINISHED,
        end_time=run1.info.end_time,
        run_name="new_run_name1",
    )
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.`run name` = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.`Run name` = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.`Run Name` = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]

    # TODO: Test attribute-based search after set_tag

    # Test run name filter works for runs logged in MLflow <= 1.29.0
    run_meta_path = Path(store.root_directory, exp_id, run1.info.run_id, "meta.yaml")
    without_run_name = run_meta_path.read_text().replace("run_name: new_run_name1\n", "")
    run_meta_path.write_text(without_run_name)
    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]
    result = store.search_runs(
        [exp_id],
        filter_string="tags.`mlflow.runName` = 'new_run_name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run1.info.run_id]


def test_search_runs_run_id(store):
    exp_id = store.create_experiment("test_search_runs_run_id")
    # Set start_time to ensure the search result is deterministic
    run1 = store.create_run(exp_id, user_id="user", start_time=1, tags=[], run_name="1")
    run2 = store.create_run(exp_id, user_id="user", start_time=2, tags=[], run_name="2")
    run_id1 = run1.info.run_id
    run_id2 = run2.info.run_id

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id = '{run_id1}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id != '{run_id1}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id IN ('{run_id1}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id NOT IN ('{run_id1}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string=f"run_name = '{run1.info.run_name}' AND run_id IN ('{run_id1}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id1]

    for filter_string in [
        f"attributes.run_id IN ('{run_id1}','{run_id2}')",
        f"attributes.run_id IN ('{run_id1}', '{run_id2}')",
        f"attributes.run_id IN ('{run_id1}',  '{run_id2}')",
    ]:
        result = store.search_runs(
            [exp_id], filter_string=filter_string, run_view_type=ViewType.ACTIVE_ONLY
        )
        assert [r.info.run_id for r in result] == [run_id2, run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string=f"attributes.run_id NOT IN ('{run_id1}', '{run_id2}')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert result == []


def test_search_runs_start_time_alias(store):
    exp_id = store.create_experiment("test_search_runs_start_time_alias")
    # Set start_time to ensure the search result is deterministic
    run1 = store.create_run(exp_id, user_id="user", start_time=1, tags=[], run_name="name")
    run2 = store.create_run(exp_id, user_id="user", start_time=2, tags=[], run_name="name")
    run_id1 = run1.info.run_id
    run_id2 = run2.info.run_id

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'name'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
    )
    assert [r.info.run_id for r in result] == [run_id2, run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'name'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.created ASC"],
    )
    assert [r.info.run_id for r in result] == [run_id1, run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.run_name = 'name'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.Created DESC"],
    )
    assert [r.info.run_id for r in result] == [run_id2, run_id1]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.start_time > 0",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.created > 1",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert [r.info.run_id for r in result] == [run_id2]

    result = store.search_runs(
        [exp_id],
        filter_string="attributes.Created > 2",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert result == []


def test_search_runs_datasets(store):
    exp_id = store.create_experiment("12345dataset")

    run1 = store.create_run(
        experiment_id=exp_id,
        user_id="user1",
        start_time=1,
        tags=[],
        run_name=None,
    )
    run2 = store.create_run(
        experiment_id=exp_id,
        user_id="user2",
        start_time=3,
        tags=[],
        run_name=None,
    )
    run3 = store.create_run(
        experiment_id=exp_id,
        user_id="user3",
        start_time=2,
        tags=[],
        run_name=None,
    )

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )
    dataset2 = Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    dataset3 = Dataset(
        name="name3",
        digest="digest3",
        source_type="st3",
        source="source3",
        schema="schema3",
        profile="profile3",
    )

    test_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="test")]
    train_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="train")]
    eval_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]

    inputs_run1 = [DatasetInput(dataset1, train_tag), DatasetInput(dataset2, eval_tag)]
    inputs_run2 = [DatasetInput(dataset1, train_tag), DatasetInput(dataset3, eval_tag)]
    inputs_run3 = [DatasetInput(dataset2, test_tag)]

    store.log_inputs(run1.info.run_id, inputs_run1)
    store.log_inputs(run2.info.run_id, inputs_run2)
    store.log_inputs(run3.info.run_id, inputs_run3)
    run_id1 = run1.info.run_id
    run_id2 = run2.info.run_id
    run_id3 = run3.info.run_id

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.name = 'name1'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id2, run_id1}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.digest = 'digest2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.name = 'name4'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == set()

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.context = 'train'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id2, run_id1}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.context = 'test'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.context = 'test' and dataset.name = 'name2'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="dataset.name = 'name2' and dataset.context = 'test'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.name IN ('name1', 'name2')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.digest IN ('digest1', 'digest2')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.name LIKE 'Name%'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == set()

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.name ILIKE 'Name%'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.context ILIKE 'test%'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3}

    result = store.search_runs(
        [exp_id],
        filter_string="datasets.context IN ('test', 'train')",
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    assert {r.info.run_id for r in result} == {run_id3, run_id1, run_id2}


def test_weird_param_names(store):
    WEIRD_PARAM_NAME = "this is/a weird/but valid param"
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.log_param(run_id, Param(WEIRD_PARAM_NAME, "Value"))
    run = store.get_run(run_id)
    assert run.data.params[WEIRD_PARAM_NAME] == "Value"


def test_log_param_empty_str(store):
    PARAM_NAME = "new param"
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.log_param(run_id, Param(PARAM_NAME, ""))
    run = store.get_run(run_id)
    assert run.data.params[PARAM_NAME] == ""


def test_log_param_with_newline(store):
    param_name = "new param"
    param_value = "a string\nwith multiple\nlines"
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.log_param(run_id, Param(param_name, param_value))
    run = store.get_run(run_id)
    assert run.data.params[param_name] == param_value


def test_log_param_enforces_value_immutability(store):
    param_name = "new param"
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.log_param(run_id, Param(param_name, "value1"))
    # Duplicate calls to `log_param` with the same key and value should succeed
    store.log_param(run_id, Param(param_name, "value1"))
    with pytest.raises(
        MlflowException, match="Changing param values is not allowed. Param with key="
    ) as e:
        store.log_param(run_id, Param(param_name, "value2"))
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    run = store.get_run(run_id)
    assert run.data.params[param_name] == "value1"


def test_log_param_max_length_value(store, monkeypatch):
    param_name = "new param"
    param_value = "x" * 6000
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.log_param(run_id, Param(param_name, param_value))
    run = store.get_run(run_id)
    assert run.data.params[param_name] == param_value
    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "false")
    with pytest.raises(MlflowException, match="exceeded length"):
        store.log_param(run_id, Param(param_name, "x" * 6001))

    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "true")
    store.log_param(run_id, Param(param_name, "x" * 6001))


def test_weird_metric_names(store):
    WEIRD_METRIC_NAME = "this is/a weird/but valid metric"
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.log_metric(run_id, Metric(WEIRD_METRIC_NAME, 10, 1234, 0))
    run = store.get_run(run_id)
    assert run.data.metrics[WEIRD_METRIC_NAME] == 10
    history = store.get_metric_history(run_id, WEIRD_METRIC_NAME)
    assert len(history) == 1
    metric = history[0]
    assert metric.key == WEIRD_METRIC_NAME
    assert metric.value == 10
    assert metric.timestamp == 1234


def test_weird_tag_names(store):
    WEIRD_TAG_NAME = "this is/a weird/but valid tag"
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.set_tag(run_id, RunTag(WEIRD_TAG_NAME, "Muhahaha!"))
    run = store.get_run(run_id)
    assert run.data.tags[WEIRD_TAG_NAME] == "Muhahaha!"


def test_set_experiment_tags(store):
    experiments, _, _ = _create_root(store)
    store.set_experiment_tag(FileStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag0", "value0"))
    store.set_experiment_tag(FileStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag1", "value1"))
    experiment = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    assert len(experiment.tags) == 2
    assert experiment.tags["tag0"] == "value0"
    assert experiment.tags["tag1"] == "value1"
    # test that updating a tag works
    store.set_experiment_tag(FileStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag0", "value00000"))
    experiment = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    assert experiment.tags["tag0"] == "value00000"
    assert experiment.tags["tag1"] == "value1"
    # test that setting a tag on 1 experiment does not impact another experiment.
    exp_id = None
    for exp in experiments:
        if exp != FileStore.DEFAULT_EXPERIMENT_ID:
            exp_id = exp
            break
    experiment = store.get_experiment(exp_id)
    assert len(experiment.tags) == 0
    # setting a tag on different experiments maintains different values across experiments
    store.set_experiment_tag(exp_id, ExperimentTag("tag1", "value11111"))
    experiment = store.get_experiment(exp_id)
    assert len(experiment.tags) == 1
    assert experiment.tags["tag1"] == "value11111"
    experiment = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    assert experiment.tags["tag0"] == "value00000"
    assert experiment.tags["tag1"] == "value1"
    # test can set multi-line tags
    store.set_experiment_tag(exp_id, ExperimentTag("multiline_tag", "value2\nvalue2\nvalue2"))
    experiment = store.get_experiment(exp_id)
    assert experiment.tags["multiline_tag"] == "value2\nvalue2\nvalue2"
    # test cannot set tags on deleted experiments
    store.delete_experiment(exp_id)
    with pytest.raises(MlflowException, match="must be in the 'active' lifecycle_stage"):
        store.set_experiment_tag(exp_id, ExperimentTag("should", "notset"))


def test_set_tags(store):
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    store.set_tag(run_id, RunTag("tag0", "value0"))
    store.set_tag(run_id, RunTag("tag1", "value1"))
    tags = store.get_run(run_id).data.tags
    assert tags["tag0"] == "value0"
    assert tags["tag1"] == "value1"

    # Can overwrite tags.
    store.set_tag(run_id, RunTag("tag0", "value2"))
    tags = store.get_run(run_id).data.tags
    assert tags["tag0"] == "value2"
    assert tags["tag1"] == "value1"

    # Can set multiline tags.
    store.set_tag(run_id, RunTag("multiline_tag", "value2\nvalue2\nvalue2"))
    tags = store.get_run(run_id).data.tags
    assert tags["multiline_tag"] == "value2\nvalue2\nvalue2"


def test_delete_tags(store):
    experiments, exp_data, _ = _create_root(store)
    exp_id = experiments[random_int(0, len(experiments) - 1)]
    run_id = exp_data[exp_id]["runs"][0]
    store.set_tag(run_id, RunTag("tag0", "value0"))
    store.set_tag(run_id, RunTag("tag1", "value1"))
    tags = store.get_run(run_id).data.tags
    assert tags["tag0"] == "value0"
    assert tags["tag1"] == "value1"
    store.delete_tag(run_id, "tag0")
    new_tags = store.get_run(run_id).data.tags
    assert "tag0" not in new_tags.keys()
    # test that you cannot delete tags that don't exist.
    with pytest.raises(MlflowException, match="No tag with name"):
        store.delete_tag(run_id, "fakeTag")
    # test that you cannot delete tags for nonexistent runs
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.delete_tag("random_id", "tag0")
    store.delete_run(run_id)
    # test that you cannot delete tags for deleted runs.
    assert store.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
    with pytest.raises(MlflowException, match="must be in 'active' lifecycle_stage"):
        store.delete_tag(run_id, "tag0")


def test_unicode_tag(store):
    _, exp_data, _ = _create_root(store)
    run_id = exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
    value = "𝐼 𝓈𝑜𝓁𝑒𝓂𝓃𝓁𝓎 𝓈𝓌𝑒𝒶𝓇 𝓉𝒽𝒶𝓉 𝐼 𝒶𝓂 𝓊𝓅 𝓉𝑜 𝓃𝑜 𝑔𝑜𝑜𝒹"
    store.set_tag(run_id, RunTag("message", value))
    tags = store.get_run(run_id).data.tags
    assert tags["message"] == value


def test_get_deleted_run(store):
    """
    Getting metrics/tags/params/run info should be allowed on deleted runs.
    """
    experiments, exp_data, _ = _create_root(store)
    exp_id = experiments[random_int(0, len(experiments) - 1)]
    run_id = exp_data[exp_id]["runs"][0]
    store.delete_run(run_id)
    assert store.get_run(run_id)


def test_set_deleted_run(store):
    """
    Setting metrics/tags/params/updating run info should not be allowed on deleted runs.
    """
    experiments, exp_data, _ = _create_root(store)
    exp_id = experiments[random_int(0, len(experiments) - 1)]
    run_id = exp_data[exp_id]["runs"][0]
    store.delete_run(run_id)

    assert store.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
    match = "must be in 'active' lifecycle_stage"
    with pytest.raises(MlflowException, match=match):
        store.set_tag(run_id, RunTag("a", "b"))
    with pytest.raises(MlflowException, match=match):
        store.log_metric(run_id, Metric("a", 0.0, timestamp=0, step=0))
    with pytest.raises(MlflowException, match=match):
        store.log_param(run_id, Param("a", "b"))


def test_default_experiment_attempted_deletion(store):
    _create_root(store)
    with pytest.raises(MlflowException, match="Cannot delete the default experiment"):
        store.delete_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    experiment = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    assert experiment.lifecycle_stage == LifecycleStage.ACTIVE
    test_id = store.create_experiment("test")
    store.delete_experiment(test_id)
    test_experiment = store.get_experiment(test_id)
    assert test_experiment.lifecycle_stage == LifecycleStage.DELETED


def test_malformed_experiment(store):
    exp_0 = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    assert exp_0.experiment_id == FileStore.DEFAULT_EXPERIMENT_ID

    experiments = len(store.search_experiments(view_type=ViewType.ALL))

    # delete metadata file.
    path = os.path.join(store.root_directory, str(exp_0.experiment_id), "meta.yaml")
    os.remove(path)
    with pytest.raises(MissingConfigException, match="does not exist"):
        store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)

    assert len(store.search_experiments(view_type=ViewType.ALL)) == experiments - 1


def test_malformed_run(store):
    _, exp_data, _ = _create_root(store)
    exp_0 = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    all_runs = _search(store, exp_0.experiment_id)

    all_run_ids = exp_data[exp_0.experiment_id]["runs"]
    assert len(all_runs) == len(all_run_ids)

    # delete metadata file.
    bad_run_id = exp_data[exp_0.experiment_id]["runs"][0]
    path = os.path.join(
        store.root_directory, str(exp_0.experiment_id), str(bad_run_id), "meta.yaml"
    )
    os.remove(path)
    with pytest.raises(MissingConfigException, match="does not exist"):
        store.get_run(bad_run_id)

    valid_runs = _search(store, exp_0.experiment_id)
    assert len(valid_runs) == len(all_runs) - 1

    for rid in all_run_ids:
        if rid != bad_run_id:
            store.get_run(rid)


def test_mismatching_experiment_id(store):
    exp_0 = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    assert exp_0.experiment_id == FileStore.DEFAULT_EXPERIMENT_ID

    experiments = len(store.search_experiments(view_type=ViewType.ALL))

    # mv experiment folder
    target = "1"
    path_orig = os.path.join(store.root_directory, str(exp_0.experiment_id))
    path_new = os.path.join(store.root_directory, str(target))
    os.rename(path_orig, path_new)

    with pytest.raises(MlflowException, match="Could not find experiment with ID"):
        store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)

    with pytest.raises(MlflowException, match="does not exist"):
        store.get_experiment(target)
    assert len(store.search_experiments(view_type=ViewType.ALL)) == experiments - 1


def test_bad_experiment_id_recorded_for_run(store):
    _, exp_data, _ = _create_root(store)
    exp_0 = store.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
    all_runs = _search(store, exp_0.experiment_id)

    all_run_ids = exp_data[exp_0.experiment_id]["runs"]
    assert len(all_runs) == len(all_run_ids)

    # change experiment pointer in run
    bad_run_id = str(exp_data[exp_0.experiment_id]["runs"][0])
    path = os.path.join(store.root_directory, str(exp_0.experiment_id), bad_run_id)
    experiment_data = read_yaml(path, "meta.yaml")
    experiment_data["experiment_id"] = 1
    write_yaml(path, "meta.yaml", experiment_data, True)

    with pytest.raises(MlflowException, match="metadata is in invalid state"):
        store.get_run(bad_run_id)

    valid_runs = _search(store, exp_0.experiment_id)
    assert len(valid_runs) == len(all_runs) - 1

    for rid in all_run_ids:
        if rid != bad_run_id:
            store.get_run(rid)


def test_log_batch(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    run_id = run.info.run_id
    metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 0)]
    param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
    tag_entities = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
    store.log_batch(
        run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
    )
    _verify_logged(store, run_id, metric_entities, param_entities, tag_entities)


def test_log_batch_max_length_value(store, monkeypatch):
    param_entities = [Param("long param", "x" * 6000), Param("short param", "xyz")]
    expected_param_entities = [
        Param("long param", "x" * 6000),
        Param("short param", "xyz"),
    ]
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    store.log_batch(run.info.run_id, (), param_entities, ())
    _verify_logged(store, run.info.run_id, (), expected_param_entities, ())

    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "false")
    param_entities = [Param("long param", "x" * 6001), Param("short param", "xyz")]
    with pytest.raises(MlflowException, match="exceeded length"):
        store.log_batch(run.info.run_id, (), param_entities, ())

    monkeypatch.setenv("MLFLOW_TRUNCATE_LONG_VALUES", "true")
    store.log_batch(run.info.run_id, (), param_entities, ())


def test_log_batch_internal_error(store):
    # Verify that internal errors during log_batch result in MlflowExceptions
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )

    def _raise_exception_fn(*args, **kwargs):
        raise Exception("Some internal error")

    with mock.patch(
        FILESTORE_PACKAGE + ".FileStore._log_run_metric"
    ) as log_metric_mock, mock.patch(
        FILESTORE_PACKAGE + ".FileStore._log_run_param"
    ) as log_param_mock, mock.patch(FILESTORE_PACKAGE + ".FileStore._set_run_tag") as set_tag_mock:
        log_metric_mock.side_effect = _raise_exception_fn
        log_param_mock.side_effect = _raise_exception_fn
        set_tag_mock.side_effect = _raise_exception_fn
        for kwargs in [
            {"metrics": [Metric("a", 3, 1, 0)]},
            {"params": [Param("b", "c")]},
            {"tags": [RunTag("c", "d")]},
        ]:
            log_batch_kwargs = {"metrics": [], "params": [], "tags": []}
            log_batch_kwargs.update(kwargs)
            with pytest.raises(MlflowException, match="Some internal error") as e:
                store.log_batch(run.info.run_id, **log_batch_kwargs)
            assert e.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_log_batch_nonexistent_run(store):
    nonexistent_uuid = uuid.uuid4().hex
    with pytest.raises(MlflowException, match=f"Run '{nonexistent_uuid}' not found") as e:
        store.log_batch(nonexistent_uuid, [], [], [])
    assert e.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_log_batch_params_idempotency(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    params = [Param("p-key", "p-val")]
    store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
    store.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
    _verify_logged(store, run.info.run_id, metrics=[], params=params, tags=[])


def test_log_batch_tags_idempotency(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])


def test_log_batch_allows_tag_overwrite(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])


def test_log_batch_same_metric_repeated_single_req(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
    metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
    store.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])


def test_log_batch_same_metric_repeated_multiple_reqs(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
    metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
    store.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0], tags=[])
    store.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
    _verify_logged(store, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])


def test_log_batch_allows_tag_overwrite_single_req(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[tags[-1]])


def test_log_batch_accepts_empty_payload(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    store.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[])


def test_log_batch_with_duplicate_params_errors_no_partial_write(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    with pytest.raises(MlflowException, match="Duplicate parameter keys have been submitted") as e:
        store.log_batch(
            run.info.run_id, metrics=[], params=[Param("a", "1"), Param("a", "2")], tags=[]
        )
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    _verify_logged(store, run.info.run_id, metrics=[], params=[], tags=[])


def test_update_run_name(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    run_id = run.info.run_id

    assert run.info.run_name == "name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "name"

    store.update_run_info(run_id, RunStatus.FINISHED, 100, "new name")
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "new name"

    store.update_run_info(run_id, RunStatus.FINISHED, 100, None)
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "new name"

    store.delete_tag(run_id, MLFLOW_RUN_NAME)
    run = store.get_run(run_id)
    assert run.info.run_name == "new name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) is None

    store.update_run_info(run_id, RunStatus.FINISHED, 100, "another name")
    run = store.get_run(run_id)
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "another name"
    assert run.info.run_name == "another name"

    store.set_tag(run_id, RunTag(MLFLOW_RUN_NAME, "yet another name"))
    run = store.get_run(run_id)
    assert run.info.run_name == "yet another name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "yet another name"

    store.log_batch(run_id, metrics=[], params=[], tags=[RunTag(MLFLOW_RUN_NAME, "batch name")])
    run = store.get_run(run_id)
    assert run.info.run_name == "batch name"
    assert run.data.tags.get(MLFLOW_RUN_NAME) == "batch name"


def test_get_metric_history_on_non_existent_metric_key(store):
    run = store.create_run(
        experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
        user_id="user",
        start_time=0,
        tags=[],
        run_name="name",
    )
    run_id = run.info.run_id
    test_metrics = store.get_metric_history(run_id, "test_metric")
    assert isinstance(test_metrics, PagedList)
    assert test_metrics == []


def test_experiment_with_default_root_artifact_uri(tmp_path):
    file_store_root_uri = path_to_local_file_uri(tmp_path)
    file_store = FileStore(file_store_root_uri)
    experiment_id = file_store.create_experiment(name="test", artifact_location="test")
    experiment_info = file_store.get_experiment(experiment_id)
    if is_windows():
        assert experiment_info.artifact_location == Path.cwd().joinpath("test").as_uri()
    else:
        assert experiment_info.artifact_location == str(Path.cwd().joinpath("test"))


def test_experiment_with_relative_artifact_uri(tmp_path):
    file_store_root_uri = append_to_uri_path(path_to_local_file_uri(tmp_path), "experiments")
    artifacts_root_uri = append_to_uri_path(path_to_local_file_uri(tmp_path), "artifacts")
    file_store = FileStore(file_store_root_uri, artifacts_root_uri)
    experiment_id = file_store.create_experiment(name="test")
    experiment_info = file_store.get_experiment(experiment_id)
    assert experiment_info.artifact_location == append_to_uri_path(
        artifacts_root_uri, experiment_id
    )


def _assert_create_run_appends_to_artifact_uri_path_correctly(
    artifact_root_uri, expected_artifact_uri_format
):
    with TempDir() as tmp:
        fs = FileStore(tmp.path(), artifact_root_uri)
        exp_id = fs.create_experiment("exp")
        run = fs.create_run(
            experiment_id=exp_id, user_id="user", start_time=0, tags=[], run_name="name"
        )
        cwd = Path.cwd().as_posix()
        drive = Path.cwd().drive
        if is_windows() and expected_artifact_uri_format.startswith("file:"):
            cwd = f"/{cwd}"
            drive = f"{drive}/"
        assert run.info.artifact_uri == expected_artifact_uri_format.format(
            e=exp_id, r=run.info.run_id, cwd=cwd, drive=drive
        )


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        (
            "\\my_server/my_path/my_sub_path",
            "file:///{drive}my_server/my_path/my_sub_path/{e}/{r}/artifacts",
        ),
        ("path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        ("/path/to/local/folder", "file:///{drive}path/to/local/folder/{e}/{r}/artifacts"),
        ("#path/to/local/folder?", "file://{cwd}/{e}/{r}/artifacts#path/to/local/folder?"),
        (
            "file:///path/to/local/folder",
            "file:///{drive}path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///{drive}path/to/local/folder/{e}/{r}/artifacts?param=value#fragment",
        ),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts?param=value",
        ),
    ],
)
def test_create_run_appends_to_artifact_local_path_file_uri_correctly_on_windows(
    input_uri, expected_uri
):
    _assert_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("path/to/local/folder", "{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        ("/path/to/local/folder", "/path/to/local/folder/{e}/{r}/artifacts"),
        ("#path/to/local/folder?", "{cwd}/#path/to/local/folder?/{e}/{r}/artifacts"),
        (
            "file:///path/to/local/folder",
            "file:///path/to/local/folder/{e}/{r}/artifacts",
        ),
        (
            "file:///path/to/local/folder?param=value#fragment",
            "file:///path/to/local/folder/{e}/{r}/artifacts?param=value#fragment",
        ),
        ("file:path/to/local/folder", "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts"),
        (
            "file:path/to/local/folder?param=value",
            "file://{cwd}/path/to/local/folder/{e}/{r}/artifacts?param=value",
        ),
    ],
)
def test_create_run_appends_to_artifact_local_path_file_uri_correctly(input_uri, expected_uri):
    _assert_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


@pytest.mark.parametrize(
    ("input_uri", "expected_uri"),
    [
        ("s3://bucket/path/to/root", "s3://bucket/path/to/root/{e}/{r}/artifacts"),
        (
            "s3://bucket/path/to/root?creds=mycreds",
            "s3://bucket/path/to/root/{e}/{r}/artifacts?creds=mycreds",
        ),
        (
            "dbscheme+driver://root@host/dbname?creds=mycreds#myfragment",
            "dbscheme+driver://root@host/dbname/{e}/{r}/artifacts?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/{e}/{r}/artifacts"
            "?creds=mycreds#myfragment",
        ),
        (
            "dbscheme+driver://root:password@hostname.com/mydb?creds=mycreds#myfragment",
            "dbscheme+driver://root:password@hostname.com/mydb/{e}/{r}/artifacts"
            "?creds=mycreds#myfragment",
        ),
    ],
)
def test_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri):
    _assert_create_run_appends_to_artifact_uri_path_correctly(input_uri, expected_uri)


def _assert_create_experiment_appends_to_artifact_uri_path_correctly(
    artifact_root_uri, expected_artifact_uri_format
):
    with TempDir() as tmp:
        fs = FileStore(tmp.path(), artifact_root_uri)
        exp_id = fs.create_experiment("exp")
        exp = fs.get_experiment(exp_id)
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
        ("\\my_server/my_path/my_sub_path", "file:///{drive}my_server/my_path/my_sub_path/{e}"),
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


def assert_dataset_inputs_equal(inputs1: List[DatasetInput], inputs2: List[DatasetInput]):
    inputs1 = sorted(inputs1, key=lambda inp: (inp.dataset.name, inp.dataset.digest))
    inputs2 = sorted(inputs2, key=lambda inp: (inp.dataset.name, inp.dataset.digest))
    assert len(inputs1) == len(inputs2)
    for idx, inp1 in enumerate(inputs1):
        inp2 = inputs2[idx]
        assert dict(inp1.dataset) == dict(inp2.dataset)
        tags1 = sorted(inp1.tags, key=lambda tag: tag.key)
        tags2 = sorted(inp2.tags, key=lambda tag: tag.key)
        for idx, tag1 in enumerate(tags1):
            tag2 = tags2[idx]
            assert tag1.key == tag1.key
            assert tag1.value == tag2.value


def test_log_inputs_and_retrieve_runs_behaves_as_expected(store):
    exp_id = store.create_experiment("12345dataset")

    run1 = store.create_run(
        experiment_id=exp_id,
        user_id="user1",
        start_time=1,
        tags=[],
        run_name=None,
    )
    run2 = store.create_run(
        experiment_id=exp_id,
        user_id="user2",
        start_time=3,
        tags=[],
        run_name=None,
    )
    run3 = store.create_run(
        experiment_id=exp_id,
        user_id="user3",
        start_time=2,
        tags=[],
        run_name=None,
    )

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )
    dataset2 = Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    dataset3 = Dataset(
        name="name3",
        digest="digest3",
        source_type="st3",
        source="source3",
        schema="schema3",
        profile="profile3",
    )

    tags1 = [InputTag(key="key1", value="value1"), InputTag(key="key2", value="value2")]
    tags2 = [InputTag(key="key3", value="value3"), InputTag(key="key4", value="value4")]
    tags3 = [InputTag(key="key5", value="value5"), InputTag(key="key6", value="value6")]

    inputs_run1 = [DatasetInput(dataset1, tags1), DatasetInput(dataset2, tags1)]
    inputs_run2 = [DatasetInput(dataset1, tags2), DatasetInput(dataset3, tags3)]
    inputs_run3 = [DatasetInput(dataset2, tags3)]

    store.log_inputs(run1.info.run_id, inputs_run1)
    store.log_inputs(run2.info.run_id, inputs_run2)
    store.log_inputs(run3.info.run_id, inputs_run3)

    run1 = store.get_run(run1.info.run_id)
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
    run2 = store.get_run(run2.info.run_id)
    assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
    run3 = store.get_run(run3.info.run_id)
    assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

    search_results_1 = store.search_runs(
        [exp_id], None, ViewType.ALL, max_results=4, order_by=["start_time ASC"]
    )
    run1 = search_results_1[0]
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
    run2 = search_results_1[2]
    assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
    run3 = search_results_1[1]
    assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)

    search_results_2 = store.search_runs(
        [exp_id], None, ViewType.ALL, max_results=4, order_by=["start_time DESC"]
    )
    run1 = search_results_2[2]
    assert_dataset_inputs_equal(run1.inputs.dataset_inputs, inputs_run1)
    run2 = search_results_2[0]
    assert_dataset_inputs_equal(run2.inputs.dataset_inputs, inputs_run2)
    run3 = search_results_2[1]
    assert_dataset_inputs_equal(run3.inputs.dataset_inputs, inputs_run3)


def test_log_input_multiple_times_does_not_overwrite_tags_or_dataset(store):
    exp_id = store.create_experiment("dataset_no_overwrite")

    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name=None,
    )
    dataset = Dataset(
        name="name",
        digest="digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    tags = [InputTag(key="key1", value="value1"), InputTag(key="key2", value="value2")]
    store.log_inputs(run.info.run_id, [DatasetInput(dataset, tags)])

    for i in range(3):
        # Since the dataset name and digest are the same as the previously logged dataset,
        # no changes should be made
        overwrite_dataset = Dataset(
            name="name",
            digest="digest",
            source_type=f"st{i}",
            source=f"source{i}",
            schema=f"schema{i}",
            profile=f"profile{i}",
        )
        # Since the dataset has already been logged as an input to the run, no changes should be
        # made to the input tags
        overwrite_tags = [
            InputTag(key=f"key{i}", value=f"value{i}"),
            InputTag(key=f"key{i+1}", value=f"value{i+1}"),
        ]
        store.log_inputs(run.info.run_id, [DatasetInput(overwrite_dataset, overwrite_tags)])

    run = store.get_run(run.info.run_id)
    assert_dataset_inputs_equal(run.inputs.dataset_inputs, [DatasetInput(dataset, tags)])

    # Logging a dataset with a different name or digest to the original run should result
    # in the addition of another dataset input
    other_name_dataset = Dataset(
        name="other_name",
        digest="digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    other_name_input_tags = [InputTag(key="k1", value="v1")]
    store.log_inputs(run.info.run_id, [DatasetInput(other_name_dataset, other_name_input_tags)])

    other_digest_dataset = Dataset(
        name="name",
        digest="other_digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    other_digest_input_tags = [InputTag(key="k2", value="v2")]
    store.log_inputs(run.info.run_id, [DatasetInput(other_digest_dataset, other_digest_input_tags)])

    run = store.get_run(run.info.run_id)
    assert_dataset_inputs_equal(
        run.inputs.dataset_inputs,
        [
            DatasetInput(dataset, tags),
            DatasetInput(other_name_dataset, other_name_input_tags),
            DatasetInput(other_digest_dataset, other_digest_input_tags),
        ],
    )

    # Logging the same dataset with different tags to new runs should result in each run
    # having its own new input tags and the same dataset input
    for i in range(3):
        new_run = store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=0,
            tags=[],
            run_name=None,
        )
        new_tags = [
            InputTag(key=f"key{i}", value=f"value{i}"),
            InputTag(key=f"key{i+1}", value=f"value{i+1}"),
        ]
        store.log_inputs(new_run.info.run_id, [DatasetInput(dataset, new_tags)])
        new_run = store.get_run(new_run.info.run_id)
        assert_dataset_inputs_equal(
            new_run.inputs.dataset_inputs, [DatasetInput(dataset, new_tags)]
        )


def test_log_inputs_uses_expected_input_and_dataset_ids_for_storage(store):
    """
    This test verifies that the FileStore uses expected IDs as folder names to represent datasets
    and run inputs. This is very important because the IDs are used to deduplicate inputs and
    datasets if the same dataset is logged to multiple runs or the same dataset is logged
    multiple times as an input to the same run with different tags.

    **If this test fails, be very careful before removing or changing asserts. Unintended changes
    could result in user-visible duplication of datasets and run inputs.**
    """
    exp_id = store.create_experiment("dataset_expected_ids")

    run1 = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name=None,
    )
    run2 = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name=None,
    )

    experiment_dir = store._get_experiment_path(exp_id, assert_exists=True)
    datasets_dir = os.path.join(experiment_dir, FileStore.DATASETS_FOLDER_NAME)

    def assert_expected_dataset_storage_ids_present(storage_ids):
        assert set(os.listdir(datasets_dir)) == set(storage_ids)

    def assert_expected_input_storage_ids_present(run, dataset_storage_ids):
        run_dir = store._get_run_dir(run.info.experiment_id, run.info.run_id)
        inputs_dir = os.path.join(run_dir, FileStore.INPUTS_FOLDER_NAME)
        expected_input_storage_ids = []
        for dataset_storage_id in dataset_storage_ids:
            md5 = insecure_hash.md5(dataset_storage_id.encode("utf-8"))
            md5.update(run.info.run_id.encode("utf-8"))
            expected_input_storage_ids.append(md5.hexdigest())
        assert set(os.listdir(inputs_dir)) == set(expected_input_storage_ids)

    tags = [InputTag(key="key", value="value")]

    dataset1 = Dataset(
        name="name",
        digest="digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    store.log_inputs(run1.info.run_id, [DatasetInput(dataset1, tags)])
    expected_dataset1_storage_id = "efa4363cd8179759e8c7f113aebdd340"
    assert_expected_dataset_storage_ids_present([expected_dataset1_storage_id])
    assert_expected_input_storage_ids_present(run1, [expected_dataset1_storage_id])

    dataset2 = Dataset(
        name="name",
        digest="digest_other",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    expected_dataset2_storage_id = "419804e8e153199481c3e509de1fef8f"
    store.log_inputs(run2.info.run_id, [DatasetInput(dataset2)])
    assert_expected_dataset_storage_ids_present(
        [expected_dataset1_storage_id, expected_dataset2_storage_id]
    )
    assert_expected_input_storage_ids_present(run2, [expected_dataset2_storage_id])

    dataset3 = Dataset(
        name="name_other",
        digest="digest",
        source_type="st",
        source="source",
        schema="schema",
        profile="profile",
    )
    expected_dataset3_storage_id = "bc5dd0841d8898512d988fe3f984313c"
    store.log_inputs(
        run2.info.run_id,
        [DatasetInput(dataset1), DatasetInput(dataset2), DatasetInput(dataset3, tags)],
    )
    assert_expected_dataset_storage_ids_present(
        [expected_dataset1_storage_id, expected_dataset2_storage_id, expected_dataset3_storage_id]
    )
    assert_expected_input_storage_ids_present(
        run2,
        [expected_dataset1_storage_id, expected_dataset2_storage_id, expected_dataset3_storage_id],
    )


def test_log_inputs_handles_case_when_no_datasets_are_specified(store):
    exp_id = store.create_experiment("log_input_no_datasets")
    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=0,
        tags=[],
        run_name=None,
    )
    store.log_inputs(run.info.run_id)
    store.log_inputs(run.info.run_id, datasets=None)


def test_search_datasets(store):
    exp_id1 = store.create_experiment("test_search_datasets_1")
    # Create an additional experiment to ensure we filter on specified experiment
    # and search works on multiple experiments.
    exp_id2 = store.create_experiment("test_search_datasets_2")

    run1 = store.create_run(
        experiment_id=exp_id1,
        user_id="user",
        start_time=1,
        tags=[],
        run_name=None,
    )
    run2 = store.create_run(
        experiment_id=exp_id1,
        user_id="user",
        start_time=2,
        tags=[],
        run_name=None,
    )
    run3 = store.create_run(
        experiment_id=exp_id2,
        user_id="user",
        start_time=3,
        tags=[],
        run_name=None,
    )

    dataset1 = Dataset(
        name="name1",
        digest="digest1",
        source_type="st1",
        source="source1",
        schema="schema1",
        profile="profile1",
    )
    dataset2 = Dataset(
        name="name2",
        digest="digest2",
        source_type="st2",
        source="source2",
        schema="schema2",
        profile="profile2",
    )
    dataset3 = Dataset(
        name="name3",
        digest="digest3",
        source_type="st3",
        source="source3",
        schema="schema3",
        profile="profile3",
    )
    dataset4 = Dataset(
        name="name4",
        digest="digest4",
        source_type="st4",
        source="source4",
        schema="schema4",
        profile="profile4",
    )

    test_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="test")]
    train_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="train")]
    eval_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]
    no_context_tag = [InputTag(key="not_context", value="test")]

    inputs_run1 = [
        DatasetInput(dataset1, train_tag),
        DatasetInput(dataset2, eval_tag),
        DatasetInput(dataset4, no_context_tag),
    ]
    inputs_run2 = [
        DatasetInput(dataset1, train_tag),
        DatasetInput(dataset2, test_tag),
    ]
    inputs_run3 = [DatasetInput(dataset3, train_tag)]

    store.log_inputs(run1.info.run_id, inputs_run1)
    store.log_inputs(run2.info.run_id, inputs_run2)
    store.log_inputs(run3.info.run_id, inputs_run3)

    # Verify actual and expected results are same size and that all elements are equal.
    def assert_has_same_elements(actual_list, expected_list):
        assert len(actual_list) == len(expected_list)
        for actual in actual_list:
            # Verify the expected results list contains same element.
            isEqual = False
            for expected in expected_list:
                isEqual = actual == expected
                if isEqual:
                    break
            assert isEqual

    # Verify no results from exp_id2 are returned.
    results = store._search_datasets([exp_id1])
    expected_results = [
        _DatasetSummary(exp_id1, dataset1.name, dataset1.digest, "train"),
        _DatasetSummary(exp_id1, dataset2.name, dataset2.digest, "eval"),
        _DatasetSummary(exp_id1, dataset2.name, dataset2.digest, "test"),
        _DatasetSummary(exp_id1, dataset4.name, dataset4.digest, None),
    ]
    assert_has_same_elements(results, expected_results)

    # Verify results from both experiment are returned.
    results = store._search_datasets([exp_id1, exp_id2])
    expected_results.append(_DatasetSummary(exp_id2, dataset3.name, dataset3.digest, "train"))
    assert_has_same_elements(results, expected_results)


def test_search_datasets_returns_no_more_than_max_results(store):
    exp_id = store.create_experiment("test_search_datasets")
    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=1,
        tags=[],
        run_name=None,
    )
    inputs = []
    # We intentionally add more than 1000 datasets here to test we only return 1000.
    for i in range(1010):
        dataset = Dataset(
            name="name" + str(i),
            digest="digest" + str(i),
            source_type="st" + str(i),
            source="source" + str(i),
            schema="schema" + str(i),
            profile="profile" + str(i),
        )
        input_tag = [InputTag(key=MLFLOW_DATASET_CONTEXT, value=str(i))]
        inputs.append(DatasetInput(dataset, input_tag))

    store.log_inputs(run.info.run_id, inputs)

    results = store._search_datasets([exp_id])
    assert len(results) == 1000
