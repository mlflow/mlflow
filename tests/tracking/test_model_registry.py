"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""
import time

from unittest import mock
import os
import sys
import pytest
import shutil
import tempfile

from mlflow.entities.model_registry import RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import path_to_local_file_uri
from tests.tracking.integration_test_utils import _await_server_down_or_die, _init_server

# pylint: disable=unused-argument

# Root directory for all stores (backend or artifact stores) created during this suite
SUITE_ROOT_DIR = tempfile.mkdtemp("test_rest_tracking")
# Root directory for all artifact stores created during this suite
SUITE_ARTIFACT_ROOT_DIR = tempfile.mkdtemp(suffix="artifacts", dir=SUITE_ROOT_DIR)


def _get_sqlite_uri():
    path = path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "test-database.bd"))
    path = path[len("file://") :]

    # NB: It looks like windows and posix have different requirements on number of slashes for
    # whatever reason. Windows needs uri like 'sqlite:///C:/path/to/my/file' whereas posix expects
    # sqlite://///path/to/my/file
    prefix = "sqlite://" if sys.platform == "win32" else "sqlite:////"
    return prefix + path


# Backend store URIs to test against
BACKEND_URIS = [
    _get_sqlite_uri(),  # SqlAlchemy
]

# Map of backend URI to tuple (server URL, Process). We populate this map by constructing
# a server per backend URI
BACKEND_URI_TO_SERVER_URL_AND_PROC = {
    uri: _init_server(backend_uri=uri, root_artifact_uri=SUITE_ARTIFACT_ROOT_DIR)
    for uri in BACKEND_URIS
}


def pytest_generate_tests(metafunc):
    """
    Automatically parametrize each each fixture/test that depends on `backend_store_uri` with the
    list of backend store URIs.
    """
    if "backend_store_uri" in metafunc.fixturenames:
        metafunc.parametrize("backend_store_uri", BACKEND_URIS)


@pytest.fixture(scope="module", autouse=True)
def server_urls():
    """
    Clean up all servers created for testing in `pytest_generate_tests`
    """
    yield
    for server_url, process in BACKEND_URI_TO_SERVER_URL_AND_PROC.values():
        print("Terminating server at %s..." % (server_url))
        print("type = ", type(process))
        process.terminate()
        _await_server_down_or_die(process)
    shutil.rmtree(SUITE_ROOT_DIR)


@pytest.fixture()
def tracking_server_uri(backend_store_uri):
    url, _ = BACKEND_URI_TO_SERVER_URL_AND_PROC[backend_store_uri]
    return url


@pytest.fixture()
def mlflow_client(tracking_server_uri):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    return mock.Mock(wraps=MlflowClient(tracking_server_uri))


def assert_is_between(start_time, end_time, expected_time):
    assert expected_time >= start_time
    assert expected_time <= end_time


def now():
    return int(time.time() * 1000)


def test_create_and_query_registered_model_flow(mlflow_client, backend_store_uri):
    name = "CreateRMTest"
    tags = {"key": "value", "another key": "some other value", "numeric value": 12345}
    start_time = now()
    registered_model = mlflow_client.create_registered_model(name, tags)
    end_time = now()
    assert isinstance(registered_model, RegisteredModel)
    assert registered_model.name == name
    assert registered_model.tags == {
        "key": "value",
        "another key": "some other value",
        "numeric value": "12345",
    }
    registered_model_detailed = mlflow_client.get_registered_model(name)
    assert isinstance(registered_model_detailed, RegisteredModel)
    assert registered_model_detailed.name == name
    assert registered_model_detailed.tags == {
        "key": "value",
        "another key": "some other value",
        "numeric value": "12345",
    }
    assert str(registered_model_detailed.description) == ""
    assert registered_model_detailed.latest_versions == []
    assert_is_between(start_time, end_time, registered_model_detailed.creation_timestamp)
    assert_is_between(start_time, end_time, registered_model_detailed.last_updated_timestamp)
    assert [name] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]
    assert [name] == [rm.name for rm in mlflow_client.search_registered_models() if rm.name == name]
    assert [name] == [
        rm.name
        for rm in mlflow_client.search_registered_models(filter_string="")
        if rm.name == name
    ]
    assert [name] == [
        rm.name
        for rm in mlflow_client.search_registered_models("name = 'CreateRMTest'")
        if rm.name == name
    ]
    # clean up test
    mlflow_client.delete_registered_model(name)


def _verify_pagination(rm_getter_with_token, expected_rms):
    result_rms = []
    result = rm_getter_with_token(None)
    result_rms.extend(result)
    first_page_size = len(result)
    while result.token:
        result = rm_getter_with_token(result.token)
        result_rms.extend(result)
        assert len(result) == first_page_size or result.token is ""
    assert [rm.name for rm in expected_rms] == [rm.name for rm in result_rms]


@pytest.mark.parametrize("max_results", [1, 6, 100])
def test_list_registered_model_flow_paginated(mlflow_client, backend_store_uri, max_results):
    names = ["CreateRMlist{:03}".format(i) for i in range(20)]
    rms = [mlflow_client.create_registered_model(name) for name in names]
    for rm in rms:
        assert isinstance(rm, RegisteredModel)

    try:
        _verify_pagination(
            lambda tok: mlflow_client.list_registered_models(
                max_results=max_results, page_token=tok
            ),
            rms,
        )
    except Exception as e:
        raise e
    finally:
        # clean up test
        for name in names:
            mlflow_client.delete_registered_model(name)


@pytest.mark.parametrize("max_results", [1, 8, 100])
@pytest.mark.parametrize(
    ("filter_string", "filter_func"),
    [
        (None, lambda rm: True),
        ("", lambda rm: True),
        ("name LIKE '%7'", lambda rm: rm.name.endswith("7")),
        ("name ILIKE '%rm%00%'", lambda rm: "00" in rm.name),
        ("name LIKE '%rm%00%'", lambda rm: False),
        ("name = 'badname'", lambda rm: False),
        ("name = 'CreateRMsearch023'", lambda rm: rm.name == "CreateRMsearch023"),
    ],
)
def test_search_registered_model_flow_paginated(
    mlflow_client, backend_store_uri, max_results, filter_string, filter_func
):
    names = ["CreateRMsearch{:03}".format(i) for i in range(29)]
    rms = [mlflow_client.create_registered_model(name) for name in names]
    for rm in rms:
        assert isinstance(rm, RegisteredModel)

    def verify_pagination(rm_getter_with_token, expected_rms):
        result_rms = []
        result = rm_getter_with_token(None)
        result_rms.extend(result)
        while result.token:
            result = rm_getter_with_token(result.token)
            result_rms.extend(result)
        assert [rm.name for rm in expected_rms] == [rm.name for rm in result_rms]

    try:
        verify_pagination(
            lambda tok: mlflow_client.search_registered_models(
                filter_string=filter_string, max_results=max_results, page_token=tok
            ),
            filter(filter_func, rms),
        )
    except Exception as e:
        raise e
    finally:
        # clean up test
        for name in names:
            mlflow_client.delete_registered_model(name)


def test_update_registered_model_flow(mlflow_client, backend_store_uri):
    name = "UpdateRMTest"
    start_time_1 = now()
    mlflow_client.create_registered_model(name)
    end_time_1 = now()
    registered_model_detailed_1 = mlflow_client.get_registered_model(name)
    assert registered_model_detailed_1.name == name
    assert str(registered_model_detailed_1.description) == ""
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.last_updated_timestamp)

    # update with no args is an error
    with pytest.raises(
        MlflowException, match="Attempting to update registered model with no new field values"
    ):
        mlflow_client.update_registered_model(name=name, description=None)

    # update name
    new_name = "UpdateRMTest 2"
    start_time_2 = now()
    mlflow_client.rename_registered_model(name=name, new_name=new_name)
    end_time_2 = now()
    with pytest.raises(MlflowException, match="Registered Model with name=UpdateRMTest not found"):
        mlflow_client.get_registered_model(name)
    registered_model_detailed_2 = mlflow_client.get_registered_model(new_name)
    assert registered_model_detailed_2.name == new_name
    assert str(registered_model_detailed_2.description) == ""
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_2.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.last_updated_timestamp)

    # update description
    start_time_3 = now()
    mlflow_client.update_registered_model(name=new_name, description="This is a test")
    end_time_3 = now()
    registered_model_detailed_3 = mlflow_client.get_registered_model(new_name)
    assert registered_model_detailed_3.name == new_name
    assert registered_model_detailed_3.description == "This is a test"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_3.creation_timestamp)
    assert_is_between(start_time_3, end_time_3, registered_model_detailed_3.last_updated_timestamp)

    # update name and description
    another_new = "UpdateRMTest 4"
    start_time_4 = now()
    mlflow_client.update_registered_model(new_name, "4th update")
    mlflow_client.rename_registered_model(new_name, another_new)
    end_time_4 = now()
    registered_model_detailed_4 = mlflow_client.get_registered_model(another_new)
    assert registered_model_detailed_4.name == another_new
    assert registered_model_detailed_4.description == "4th update"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_4.creation_timestamp)
    assert_is_between(start_time_4, end_time_4, registered_model_detailed_4.last_updated_timestamp)

    # using rename
    previous_name = another_new
    another_new = "UpdateRMTest 5"
    start_time_5 = now()
    mlflow_client.rename_registered_model(previous_name, another_new)
    end_time_5 = now()
    registered_model_detailed_5 = mlflow_client.get_registered_model(another_new)
    assert registered_model_detailed_5.name == another_new
    assert registered_model_detailed_5.description == "4th update"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_5.creation_timestamp)
    assert_is_between(start_time_5, end_time_5, registered_model_detailed_5.last_updated_timestamp)

    # old named models are not accessible
    for old_name in [previous_name, name, new_name]:
        with pytest.raises(
            MlflowException, match=r"Registered Model with name=UpdateRMTest( \d)? not found"
        ):
            mlflow_client.get_registered_model(old_name)


def test_delete_registered_model_flow(mlflow_client, backend_store_uri):
    name = "DeleteRMTest"
    start_time_1 = now()
    mlflow_client.create_registered_model(name)
    end_time_1 = now()
    registered_model_detailed_1 = mlflow_client.get_registered_model(name)
    assert registered_model_detailed_1.name == name
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.last_updated_timestamp)

    assert [name] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]

    # cannot create a model with same name
    with pytest.raises(MlflowException, match=r"Registered Model .+ already exists"):
        mlflow_client.create_registered_model(name)

    mlflow_client.delete_registered_model(name)

    # cannot get a deleted model
    with pytest.raises(MlflowException, match=r"Registered Model .+ not found"):
        mlflow_client.get_registered_model(name)

    # cannot update a deleted model
    with pytest.raises(MlflowException, match=r"Registered Model .+ not found"):
        mlflow_client.rename_registered_model(name=name, new_name="something else")

    # list does not include deleted model
    assert [] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]

    # recreate model with same name
    start_time_2 = now()
    mlflow_client.create_registered_model(name)
    end_time_2 = now()
    registered_model_detailed_2 = mlflow_client.get_registered_model(name)
    assert registered_model_detailed_2.name == name
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.last_updated_timestamp)

    assert [name] == [rm.name for rm in mlflow_client.list_registered_models() if rm.name == name]


def test_set_delete_registered_model_tag_flow(mlflow_client, backend_store_uri):
    name = "SetDeleteRMTagTest"
    mlflow_client.create_registered_model(name)
    registered_model_detailed = mlflow_client.get_registered_model(name)
    assert registered_model_detailed.tags == {}
    tags = {"key": "value", "numeric value": 12345}
    for key, value in tags.items():
        mlflow_client.set_registered_model_tag(name, key, value)
    registered_model_detailed = mlflow_client.get_registered_model(name)
    assert registered_model_detailed.tags == {"key": "value", "numeric value": "12345"}
    mlflow_client.delete_registered_model_tag(name, "key")
    registered_model_detailed = mlflow_client.get_registered_model(name)
    assert registered_model_detailed.tags == {"numeric value": "12345"}


def test_create_and_query_model_version_flow(mlflow_client, backend_store_uri):
    name = "CreateMVTest"
    tags = {"key": "value", "another key": "some other value", "numeric value": 12345}
    mlflow_client.create_registered_model(name)
    mv1 = mlflow_client.create_model_version(name, "path/to/model", "run_id_1", tags)
    assert mv1.version == "1"
    assert mv1.name == name
    assert mv1.tags == {"key": "value", "another key": "some other value", "numeric value": "12345"}
    mvd1 = mlflow_client.get_model_version(name, "1")
    assert mvd1.tags == {
        "key": "value",
        "another key": "some other value",
        "numeric value": "12345",
    }
    assert [[mvd1]] == [
        rm.latest_versions for rm in mlflow_client.list_registered_models() if rm.name == name
    ]
    mv2 = mlflow_client.create_model_version(name, "another_path/to/model", "run_id_1")
    assert mv2.version == "2"
    assert mv2.name == name
    mvd2 = mlflow_client.get_model_version(name, "2")
    assert [[mvd2]] == [
        rm.latest_versions for rm in mlflow_client.list_registered_models() if rm.name == name
    ]
    model_versions_by_name = mlflow_client.search_model_versions("name = '%s'" % name)
    assert set(["1", "2"]) == set([mv.version for mv in model_versions_by_name])
    assert set([name]) == set([mv.name for mv in model_versions_by_name])

    mv3 = mlflow_client.create_model_version(name, "another_path/to/model", "run_id_2")
    assert mv3.version == "3"
    assert [mvd1] == mlflow_client.search_model_versions("source_path = 'path/to/model'")
    assert [mvd1, mvd2] == mlflow_client.search_model_versions("run_id = 'run_id_1'")

    assert "path/to/model" == mlflow_client.get_model_version_download_uri(name, "1")


def test_get_model_version(mlflow_client, backend_store_uri):
    name = "GetModelVersionTest"
    mlflow_client.create_registered_model(name)
    mlflow_client.create_model_version(name, "path/to/model", "run_id_1")
    model_version = mlflow_client.get_model_version(name, "1")
    assert model_version.name == name
    assert model_version.version == "1"

    with pytest.raises(
        MlflowException, match="INVALID_PARAMETER_VALUE: Model version must be an integer"
    ):
        mlflow_client.get_model_version(name=name, version="something not correct")


def test_update_model_version_flow(mlflow_client, backend_store_uri):
    name = "UpdateMVTest"
    start_time_0 = now()
    mlflow_client.create_registered_model(name)
    end_time_0 = now()
    rmd1 = mlflow_client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd1.creation_timestamp)
    assert_is_between(start_time_0, end_time_0, rmd1.last_updated_timestamp)

    start_time_1 = now()
    mv1 = mlflow_client.create_model_version(name, "path/to/model", "run_id_1")
    end_time_1 = now()
    assert mv1.version == "1"
    assert mv1.name == name
    mvd1 = mlflow_client.get_model_version(name, "1")
    assert str(mvd1.description) == ""
    assert_is_between(start_time_1, end_time_1, mvd1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, mvd1.last_updated_timestamp)

    # creating model version changes last_updated_timestamp for registered model
    rmd2 = mlflow_client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd2.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, rmd2.last_updated_timestamp)

    assert [[mvd1]] == [
        rm.latest_versions for rm in mlflow_client.list_registered_models() if rm.name == name
    ]
    mv2 = mlflow_client.create_model_version(name, "another_path/to/model", "run_id_1")
    assert mv2.version == "2"
    assert mv2.name == name
    mvd2 = mlflow_client.get_model_version(name, "2")
    assert [[mvd2]] == [
        rm.latest_versions for rm in mlflow_client.list_registered_models() if rm.name == name
    ]

    start_time_2 = now()
    mlflow_client.transition_model_version_stage(name=name, version=1, stage="Staging")
    end_time_2 = now()
    mvd1b = mlflow_client.get_model_version(name, 1)
    assert_is_between(start_time_1, end_time_1, mvd1b.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, mvd1b.last_updated_timestamp)

    # updating model version's stage changes last_updated_timestamp for registered model
    rmd3 = mlflow_client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd3.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, rmd3.last_updated_timestamp)

    model_versions_detailed = [
        rm.latest_versions for rm in mlflow_client.list_registered_models() if rm.name == name
    ]
    assert 1 == len(model_versions_detailed)
    assert set(["1", "2"]) == set([mvd.version for mvd in model_versions_detailed[0]])
    assert set([name]) == set([mvd.name for mvd in model_versions_detailed[0]])

    # update description
    start_time_3 = now()
    mlflow_client.update_model_version(name=name, version=1, description="This is a test model")
    end_time_3 = now()
    mvd1c = mlflow_client.get_model_version(name, "1")
    assert str(mvd1c.description) == "This is a test model"
    assert_is_between(start_time_1, end_time_1, mvd1c.creation_timestamp)
    assert_is_between(start_time_3, end_time_3, mvd1c.last_updated_timestamp)

    # changing description for model version does not affect last_updated_timestamp for registered
    # model
    rmd4 = mlflow_client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd4.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, rmd4.last_updated_timestamp)


def test_latest_models(mlflow_client, backend_store_uri):
    version_stage_mapping = (
        ("1", "Archived"),
        ("2", "Production"),
        ("3", "Archived"),
        ("4", "Production"),
        ("5", "Staging"),
        ("6", "Staging"),
        ("7", "None"),
    )
    name = "LatestVersionTest"
    mlflow_client.create_registered_model(name)

    for version, stage in version_stage_mapping:
        mv = mlflow_client.create_model_version(name, "path/to/model", "run_id")
        assert mv.version == version
        if stage != "None":
            mlflow_client.transition_model_version_stage(name, version, stage=stage)
        mvd = mlflow_client.get_model_version(name, version)
        assert mvd.current_stage == stage

    def get_latest(stages):
        latest = mlflow_client.get_latest_versions(name, stages)
        return {mvd.current_stage: mvd.version for mvd in latest}

    assert {"None": "7"} == get_latest(["None"])
    assert {"Staging": "6"} == get_latest(["Staging"])
    assert {"None": "7", "Staging": "6"} == get_latest(["None", "Staging"])
    assert {"Production": "4", "Staging": "6", "Archived": "3", "None": "7"} == get_latest(None)
    assert {"Production": "4", "Staging": "6", "Archived": "3", "None": "7"} == get_latest([])


def test_delete_model_version_flow(mlflow_client, backend_store_uri):
    name = "DeleteMVTest"
    start_time_0 = now()
    mlflow_client.create_registered_model(name)
    end_time_0 = now()
    rmd1 = mlflow_client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd1.creation_timestamp)
    assert_is_between(start_time_0, end_time_0, rmd1.last_updated_timestamp)

    start_time_1 = now()
    mv1 = mlflow_client.create_model_version(name, "path/to/model", "run_id_1")
    end_time_1 = now()
    assert mv1.version == "1"
    assert mv1.name == name
    mvd1 = mlflow_client.get_model_version(name, 1)
    assert_is_between(start_time_1, end_time_1, mvd1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, mvd1.last_updated_timestamp)

    # creating model version changes last_updated_timestamp for registered model
    rmd2 = mlflow_client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd2.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, rmd2.last_updated_timestamp)

    mv2 = mlflow_client.create_model_version(name, "another_path/to/model", "run_id_1")
    assert mv2.version == "2"
    assert mv2.name == name
    mv3 = mlflow_client.create_model_version(name, "a/b/c", "run_id_2")
    assert mv3.version == "3"
    assert mv3.name == name
    model_versions_detailed = [
        rm.latest_versions for rm in mlflow_client.list_registered_models() if rm.name == name
    ]
    assert 1 == len(model_versions_detailed)
    assert "3" == model_versions_detailed[0][0].version
    assert {"1", "2", "3"} == set(
        [mv.version for mv in mlflow_client.search_model_versions("name = '%s'" % name)]
    )

    start_time_2 = now()
    mlflow_client.delete_model_version(name, "1")
    end_time_2 = now()
    assert {"2", "3"} == set(
        [mv.version for mv in mlflow_client.search_model_versions("name = '%s'" % name)]
    )
    rmd3 = mlflow_client.get_registered_model(name)
    # deleting model versions changes last_updated_timestamp for registered model
    assert_is_between(start_time_0, end_time_0, rmd3.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, rmd3.last_updated_timestamp)

    # cannot get a deleted model version
    with pytest.raises(MlflowException, match=r"Model Version .+ not found"):
        mlflow_client.delete_model_version(name, "1")

    # cannot update a deleted model version
    with pytest.raises(MlflowException, match=r"Model Version .+ not found"):
        mlflow_client.update_model_version(name=name, version=1, description="Test model")
    with pytest.raises(MlflowException, match=r"Model Version .+ not found"):
        mlflow_client.transition_model_version_stage(name=name, version=1, stage="Staging")

    mlflow_client.delete_model_version(name, 3)
    assert {"2"} == set(
        [mv.version for mv in mlflow_client.search_model_versions("name = '%s'" % name)]
    )

    # new model versions will not reuse existing version numbers
    mv4 = mlflow_client.create_model_version(name, "a/b/c", "run_id_2")
    assert mv4.version == "4"
    assert mv4.name == name
    assert {"2", "4"} == set(
        [mv.version for mv in mlflow_client.search_model_versions("name = '%s'" % name)]
    )


def test_set_delete_model_version_tag_flow(mlflow_client, backend_store_uri):
    name = "SetDeleteMVTagTest"
    mlflow_client.create_registered_model(name)
    mlflow_client.create_model_version(name, "path/to/model", "run_id_1")
    model_version_detailed = mlflow_client.get_model_version(name, "1")
    assert model_version_detailed.tags == {}
    tags = {"key": "value", "numeric value": 12345}
    for key, value in tags.items():
        mlflow_client.set_model_version_tag(name, "1", key, value)
    model_version_detailed = mlflow_client.get_model_version(name, "1")
    assert model_version_detailed.tags == {"key": "value", "numeric value": "12345"}
    mlflow_client.delete_model_version_tag(name, "1", "key")
    model_version_detailed = mlflow_client.get_model_version(name, "1")
    assert model_version_detailed.tags == {"numeric value": "12345"}
