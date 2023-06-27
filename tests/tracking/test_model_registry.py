"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""
import time
import pytest

from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient
from mlflow.utils.time_utils import get_current_time_millis
from mlflow.utils.os import is_windows
from tests.tracking.integration_test_utils import _terminate_server, _init_server


@pytest.fixture(params=["file", "sqlalchemy"])
def client(request, tmp_path):
    if request.param == "file":
        backend_uri = tmp_path.joinpath("file").as_uri()
    else:
        path = tmp_path.joinpath("sqlalchemy.db").as_uri()
        backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    url, process = _init_server(
        backend_uri=backend_uri, root_artifact_uri=tmp_path.joinpath("artifacts").as_uri()
    )
    yield MlflowClient(url)
    _terminate_server(process)


def assert_is_between(start_time, end_time, expected_time):
    assert expected_time >= start_time
    assert expected_time <= end_time


def test_create_and_query_registered_model_flow(client):
    name = "CreateRMTest"
    tags = {"key": "value", "another key": "some other value", "numeric value": 12345}
    start_time = get_current_time_millis()
    registered_model = client.create_registered_model(name, tags)
    end_time = get_current_time_millis()
    assert isinstance(registered_model, RegisteredModel)
    assert registered_model.name == name
    assert registered_model.tags == {
        "key": "value",
        "another key": "some other value",
        "numeric value": "12345",
    }
    registered_model_detailed = client.get_registered_model(name)
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
    assert [rm.name for rm in client.search_registered_models() if rm.name == name] == [name]
    assert [rm.name for rm in client.search_registered_models() if rm.name == name] == [name]
    assert [
        rm.name for rm in client.search_registered_models(filter_string="") if rm.name == name
    ] == [name]
    assert [
        rm.name
        for rm in client.search_registered_models("name = 'CreateRMTest'")
        if rm.name == name
    ] == [name]


def _verify_pagination(rm_getter_with_token, expected_rms):
    result_rms = []
    result = rm_getter_with_token(None)
    result_rms.extend(result)
    first_page_size = len(result)
    while result.token:
        result = rm_getter_with_token(result.token)
        result_rms.extend(result)
        assert len(result) == first_page_size or result.token == ""
    assert [rm.name for rm in expected_rms] == [rm.name for rm in result_rms]


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
def test_search_registered_model_flow_paginated(client, max_results, filter_string, filter_func):
    names = [f"CreateRMsearch{i:03}" for i in range(29)]
    rms = [client.create_registered_model(name) for name in names]
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

    verify_pagination(
        lambda tok: client.search_registered_models(
            filter_string=filter_string, max_results=max_results, page_token=tok
        ),
        filter(filter_func, rms),
    )


def test_update_registered_model_flow(client):
    name = "UpdateRMTest"
    start_time_1 = get_current_time_millis()
    client.create_registered_model(name)
    end_time_1 = get_current_time_millis()
    registered_model_detailed_1 = client.get_registered_model(name)
    assert registered_model_detailed_1.name == name
    assert str(registered_model_detailed_1.description) == ""
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.last_updated_timestamp)

    # update with no args is an error
    with pytest.raises(
        MlflowException, match="Attempting to update registered model with no new field values"
    ):
        client.update_registered_model(name=name, description=None)

    # update name
    new_name = "UpdateRMTest 2"
    start_time_2 = get_current_time_millis()
    client.rename_registered_model(name=name, new_name=new_name)
    end_time_2 = get_current_time_millis()
    with pytest.raises(MlflowException, match="Registered Model with name=UpdateRMTest not found"):
        client.get_registered_model(name)
    registered_model_detailed_2 = client.get_registered_model(new_name)
    assert registered_model_detailed_2.name == new_name
    assert str(registered_model_detailed_2.description) == ""
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_2.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.last_updated_timestamp)

    # update description
    start_time_3 = get_current_time_millis()
    client.update_registered_model(name=new_name, description="This is a test")
    end_time_3 = get_current_time_millis()
    registered_model_detailed_3 = client.get_registered_model(new_name)
    assert registered_model_detailed_3.name == new_name
    assert registered_model_detailed_3.description == "This is a test"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_3.creation_timestamp)
    assert_is_between(start_time_3, end_time_3, registered_model_detailed_3.last_updated_timestamp)

    # update name and description
    another_new = "UpdateRMTest 4"
    start_time_4 = get_current_time_millis()
    client.update_registered_model(new_name, "4th update")
    client.rename_registered_model(new_name, another_new)
    end_time_4 = get_current_time_millis()
    registered_model_detailed_4 = client.get_registered_model(another_new)
    assert registered_model_detailed_4.name == another_new
    assert registered_model_detailed_4.description == "4th update"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_4.creation_timestamp)
    assert_is_between(start_time_4, end_time_4, registered_model_detailed_4.last_updated_timestamp)

    # using rename
    previous_name = another_new
    another_new = "UpdateRMTest 5"
    start_time_5 = get_current_time_millis()
    client.rename_registered_model(previous_name, another_new)
    end_time_5 = get_current_time_millis()
    registered_model_detailed_5 = client.get_registered_model(another_new)
    assert registered_model_detailed_5.name == another_new
    assert registered_model_detailed_5.description == "4th update"
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_5.creation_timestamp)
    assert_is_between(start_time_5, end_time_5, registered_model_detailed_5.last_updated_timestamp)

    # old named models are not accessible
    for old_name in [previous_name, name, new_name]:
        with pytest.raises(
            MlflowException, match=r"Registered Model with name=UpdateRMTest( \d)? not found"
        ):
            client.get_registered_model(old_name)


def test_delete_registered_model_flow(client):
    name = "DeleteRMTest"
    start_time_1 = get_current_time_millis()
    client.create_registered_model(name)
    end_time_1 = get_current_time_millis()
    registered_model_detailed_1 = client.get_registered_model(name)
    assert registered_model_detailed_1.name == name
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, registered_model_detailed_1.last_updated_timestamp)

    assert [rm.name for rm in client.search_registered_models() if rm.name == name] == [name]

    # cannot create a model with same name
    with pytest.raises(MlflowException, match=r"Registered Model .+ already exists"):
        client.create_registered_model(name)

    client.delete_registered_model(name)

    # cannot get a deleted model
    with pytest.raises(MlflowException, match=r"Registered Model .+ not found"):
        client.get_registered_model(name)

    # cannot update a deleted model
    with pytest.raises(MlflowException, match=r"Registered Model .+ not found"):
        client.rename_registered_model(name=name, new_name="something else")

    # list does not include deleted model
    assert [rm.name for rm in client.search_registered_models() if rm.name == name] == []

    # recreate model with same name
    start_time_2 = get_current_time_millis()
    client.create_registered_model(name)
    end_time_2 = get_current_time_millis()
    registered_model_detailed_2 = client.get_registered_model(name)
    assert registered_model_detailed_2.name == name
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, registered_model_detailed_2.last_updated_timestamp)

    assert [rm.name for rm in client.search_registered_models() if rm.name == name] == [name]


def test_set_delete_registered_model_tag_flow(client):
    name = "SetDeleteRMTagTest"
    client.create_registered_model(name)
    registered_model_detailed = client.get_registered_model(name)
    assert registered_model_detailed.tags == {}
    tags = {"key": "value", "numeric value": 12345}
    for key, value in tags.items():
        client.set_registered_model_tag(name, key, value)
    registered_model_detailed = client.get_registered_model(name)
    assert registered_model_detailed.tags == {"key": "value", "numeric value": "12345"}
    client.delete_registered_model_tag(name, "key")
    registered_model_detailed = client.get_registered_model(name)
    assert registered_model_detailed.tags == {"numeric value": "12345"}


def test_set_registered_model_tag_with_empty_string_as_value(client):
    name = "SetRMTagEmptyValueTest"
    client.create_registered_model(name)
    client.set_registered_model_tag(name, "tag_key", "")
    assert {"tag_key": ""}.items() <= client.get_registered_model(name).tags.items()


@pytest.mark.parametrize(
    ("filter_string", "filter_func"),
    [
        (None, lambda mv: True),
        ("", lambda mv: True),
        ("name LIKE '%2'", lambda mv: mv.name.endswith("2")),
        ("name ILIKE '%rm%00%'", lambda mv: "00" in mv.name),
        ("name LIKE '%rm%00%'", lambda mv: False),
        ("name = 'badname'", lambda mv: False),
        ("name = 'CreateRMsearchForMV03'", lambda mv: mv.name == "CreateRMsearchForMV03"),
    ],
)
def test_search_model_versions_filter_string(
    client,
    filter_string,
    filter_func,
):
    names = [f"CreateRMsearchForMV{i:03}" for i in range(29)]
    for name in names:
        client.create_registered_model(name)
    mvs = []
    for name in names + names[:10]:
        # Sleep for unique creation_time to make search results deterministic
        time.sleep(0.001)
        mvs.append(client.create_model_version(name, "runs:/run_id/model", "run_id"))
    for mv in mvs:
        assert isinstance(mv, ModelVersion)
    mvs = mvs[::-1]

    def verify_pagination(mv_getter_with_token, expected_mvs):
        result_mvs = []
        result = mv_getter_with_token(None)
        result_mvs.extend(result)
        while result.token:
            result = mv_getter_with_token(result.token)
            result_mvs.extend(result)
        assert [mv.name for mv in expected_mvs] == [mv.name for mv in result_mvs]
        assert [mv.version for mv in expected_mvs] == [mv.version for mv in result_mvs]

    expected_mvs = sorted(
        filter(filter_func, mvs), key=lambda x: x.last_updated_timestamp, reverse=True
    )
    verify_pagination(
        lambda tok: client.search_model_versions(
            filter_string=filter_string,
            page_token=tok,
        ),
        expected_mvs,
    )


@pytest.mark.parametrize("max_results", [1, 8, 100])
def test_search_model_versions_max_results(client, max_results):
    names = [f"CreateRMsearchForMV{i:03}" for i in range(29)]
    for name in names:
        client.create_registered_model(name)
    mvs = []
    for name in names + names[:10]:
        # Sleep for unique creation_time to make search results deterministic
        time.sleep(0.001)
        mvs.append(client.create_model_version(name, "runs:/run_id/model", "run_id"))
    for mv in mvs:
        assert isinstance(mv, ModelVersion)
    mvs = mvs[::-1]

    def verify_pagination(mv_getter_with_token, expected_mvs):
        result_mvs = []
        result = mv_getter_with_token(None)
        result_mvs.extend(result)
        while result.token:
            result = mv_getter_with_token(result.token)
            result_mvs.extend(result)
        assert [mv.name for mv in expected_mvs] == [mv.name for mv in result_mvs]
        assert [mv.version for mv in expected_mvs] == [mv.version for mv in result_mvs]

    expected_mvs = sorted(mvs, key=lambda x: x.last_updated_timestamp, reverse=True)
    verify_pagination(
        lambda tok: client.search_model_versions(
            max_results=max_results,
            page_token=tok,
        ),
        expected_mvs,
    )


@pytest.mark.parametrize(
    ("order_by", "order_by_key", "order_by_desc"),
    [
        (None, None, False),
        (["name DESC"], lambda mv: (mv.name, mv.version), True),
        (
            ["version_number DESC"],
            lambda mv: (-int(mv.version), mv.name),
            False,
        ),
    ],
)
def test_search_model_versions_order_by(
    client,
    order_by,
    order_by_key,
    order_by_desc,
):
    names = [f"CreateRMsearchForMV{i:03}" for i in range(29)]
    for name in names:
        client.create_registered_model(name)
    mvs = []
    for name in names + names[:10]:
        # Sleep for unique creation_time to make search results deterministic
        time.sleep(0.001)
        mvs.append(client.create_model_version(name, "runs:/run_id/model", "run_id"))
    for mv in mvs:
        assert isinstance(mv, ModelVersion)
    mvs = mvs[::-1]

    def verify_pagination(mv_getter_with_token, expected_mvs):
        result_mvs = []
        result = mv_getter_with_token(None)
        result_mvs.extend(result)
        while result.token:
            result = mv_getter_with_token(result.token)
            result_mvs.extend(result)
        assert [mv.name for mv in expected_mvs] == [mv.name for mv in result_mvs]
        assert [mv.version for mv in expected_mvs] == [mv.version for mv in result_mvs]

    if order_by_key:
        expected_mvs = sorted(mvs, key=order_by_key, reverse=order_by_desc)
    else:
        expected_mvs = sorted(
            mvs, key=lambda x: (-int(x.last_updated_timestamp), x.name, -int(x.version))
        )
    verify_pagination(
        lambda tok: client.search_model_versions(
            order_by=order_by,
            page_token=tok,
        ),
        expected_mvs,
    )


def test_create_and_query_model_version_flow(client):
    name = "CreateMVTest"
    tags = {"key": "value", "another key": "some other value", "numeric value": 12345}
    client.create_registered_model(name)
    mv1 = client.create_model_version(name, "runs:/run_id/model", "run_id_1", tags)
    assert mv1.version == "1"
    assert mv1.name == name
    assert mv1.tags == {"key": "value", "another key": "some other value", "numeric value": "12345"}
    mvd1 = client.get_model_version(name, "1")
    assert mvd1.tags == {
        "key": "value",
        "another key": "some other value",
        "numeric value": "12345",
    }
    assert [rm.latest_versions for rm in client.search_registered_models() if rm.name == name] == [
        [mvd1]
    ]
    mv2 = client.create_model_version(name, "runs:/run_id/another_model", "run_id_1")
    assert mv2.version == "2"
    assert mv2.name == name
    mvd2 = client.get_model_version(name, "2")
    assert [rm.latest_versions for rm in client.search_registered_models() if rm.name == name] == [
        [mvd2]
    ]
    model_versions_by_name = client.search_model_versions("name = '%s'" % name)
    assert {mv.version for mv in model_versions_by_name} == {"1", "2"}
    assert {mv.name for mv in model_versions_by_name} == {name}

    mv3 = client.create_model_version(name, "runs:/run_id/another_model", "run_id_2")
    assert mv3.version == "3"
    assert client.search_model_versions("source_path = 'runs:/run_id/model'") == [mvd1]
    assert client.search_model_versions("run_id = 'run_id_1'") == [mvd2, mvd1]

    assert client.get_model_version_download_uri(name, "1") == "runs:/run_id/model"


def test_get_model_version(client):
    name = "GetModelVersionTest"
    client.create_registered_model(name)
    client.create_model_version(name, "runs:/run_id/model", "run_id_1")
    model_version = client.get_model_version(name, "1")
    assert model_version.name == name
    assert model_version.version == "1"

    with pytest.raises(
        MlflowException, match="INVALID_PARAMETER_VALUE: Model version must be an integer"
    ):
        client.get_model_version(name=name, version="something not correct")


def test_update_model_version_flow(client):
    name = "UpdateMVTest"
    start_time_0 = get_current_time_millis()
    client.create_registered_model(name)
    end_time_0 = get_current_time_millis()
    rmd1 = client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd1.creation_timestamp)
    assert_is_between(start_time_0, end_time_0, rmd1.last_updated_timestamp)

    start_time_1 = get_current_time_millis()
    mv1 = client.create_model_version(name, "runs:/run_id/model", "run_id_1")
    end_time_1 = get_current_time_millis()
    assert mv1.version == "1"
    assert mv1.name == name
    mvd1 = client.get_model_version(name, "1")
    assert str(mvd1.description) == ""
    assert_is_between(start_time_1, end_time_1, mvd1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, mvd1.last_updated_timestamp)

    # creating model version changes last_updated_timestamp for registered model
    rmd2 = client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd2.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, rmd2.last_updated_timestamp)

    assert [rm.latest_versions for rm in client.search_registered_models() if rm.name == name] == [
        [mvd1]
    ]
    mv2 = client.create_model_version(name, "runs:/run_id/another_model", "run_id_1")
    assert mv2.version == "2"
    assert mv2.name == name
    mvd2 = client.get_model_version(name, "2")
    assert [rm.latest_versions for rm in client.search_registered_models() if rm.name == name] == [
        [mvd2]
    ]

    start_time_2 = get_current_time_millis()
    client.transition_model_version_stage(name=name, version=1, stage="Staging")
    end_time_2 = get_current_time_millis()
    mvd1b = client.get_model_version(name, 1)
    assert_is_between(start_time_1, end_time_1, mvd1b.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, mvd1b.last_updated_timestamp)

    # updating model version's stage changes last_updated_timestamp for registered model
    rmd3 = client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd3.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, rmd3.last_updated_timestamp)

    model_versions_detailed = [
        rm.latest_versions for rm in client.search_registered_models() if rm.name == name
    ]
    assert len(model_versions_detailed) == 1
    assert {mvd.version for mvd in model_versions_detailed[0]} == {"1", "2"}
    assert {mvd.name for mvd in model_versions_detailed[0]} == {name}

    # update description
    start_time_3 = get_current_time_millis()
    client.update_model_version(name=name, version=1, description="This is a test model")
    end_time_3 = get_current_time_millis()
    mvd1c = client.get_model_version(name, "1")
    assert str(mvd1c.description) == "This is a test model"
    assert_is_between(start_time_1, end_time_1, mvd1c.creation_timestamp)
    assert_is_between(start_time_3, end_time_3, mvd1c.last_updated_timestamp)

    # changing description for model version does not affect last_updated_timestamp for registered
    # model
    rmd4 = client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd4.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, rmd4.last_updated_timestamp)


def test_latest_models(client):
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
    client.create_registered_model(name)

    for version, stage in version_stage_mapping:
        # Sleep for unique creation_time to make search results deterministic
        time.sleep(0.001)
        mv = client.create_model_version(name, "runs:/run_id/model", "run_id")
        assert mv.version == version
        if stage != "None":
            client.transition_model_version_stage(name, version, stage=stage)
        mvd = client.get_model_version(name, version)
        assert mvd.current_stage == stage

    def get_latest(stages):
        latest = client.get_latest_versions(name, stages)
        return {mvd.current_stage: mvd.version for mvd in latest}

    assert get_latest(["None"]) == {"None": "7"}
    assert get_latest(["Staging"]) == {"Staging": "6"}
    assert get_latest(["None", "Staging"]) == {"None": "7", "Staging": "6"}
    assert get_latest(None) == {"Production": "4", "Staging": "6", "Archived": "3", "None": "7"}
    assert get_latest([]) == {"Production": "4", "Staging": "6", "Archived": "3", "None": "7"}


def test_delete_model_version_flow(client):
    name = "DeleteMVTest"
    start_time_0 = get_current_time_millis()
    client.create_registered_model(name)
    end_time_0 = get_current_time_millis()
    rmd1 = client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd1.creation_timestamp)
    assert_is_between(start_time_0, end_time_0, rmd1.last_updated_timestamp)

    start_time_1 = get_current_time_millis()
    mv1 = client.create_model_version(name, "runs:/run_id/model", "run_id_1")
    end_time_1 = get_current_time_millis()
    assert mv1.version == "1"
    assert mv1.name == name
    mvd1 = client.get_model_version(name, 1)
    assert_is_between(start_time_1, end_time_1, mvd1.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, mvd1.last_updated_timestamp)

    # creating model version changes last_updated_timestamp for registered model
    rmd2 = client.get_registered_model(name)
    assert_is_between(start_time_0, end_time_0, rmd2.creation_timestamp)
    assert_is_between(start_time_1, end_time_1, rmd2.last_updated_timestamp)

    mv2 = client.create_model_version(name, "runs:/run_id/another_model", "run_id_1")
    assert mv2.version == "2"
    assert mv2.name == name
    mv3 = client.create_model_version(name, "runs:/run_id_2/a/b/c", "run_id_2")
    assert mv3.version == "3"
    assert mv3.name == name
    model_versions_detailed = [
        rm.latest_versions for rm in client.search_registered_models() if rm.name == name
    ]
    assert len(model_versions_detailed) == 1
    assert model_versions_detailed[0][0].version == "3"
    assert {mv.version for mv in client.search_model_versions("name = '%s'" % name)} == {
        "1",
        "2",
        "3",
    }

    start_time_2 = get_current_time_millis()
    client.delete_model_version(name, "1")
    end_time_2 = get_current_time_millis()
    assert {mv.version for mv in client.search_model_versions("name = '%s'" % name)} == {
        "2",
        "3",
    }
    rmd3 = client.get_registered_model(name)
    # deleting model versions changes last_updated_timestamp for registered model
    assert_is_between(start_time_0, end_time_0, rmd3.creation_timestamp)
    assert_is_between(start_time_2, end_time_2, rmd3.last_updated_timestamp)

    # cannot get a deleted model version
    with pytest.raises(MlflowException, match=r"Model Version .+ not found"):
        client.delete_model_version(name, "1")

    # cannot update a deleted model version
    with pytest.raises(MlflowException, match=r"Model Version .+ not found"):
        client.update_model_version(name=name, version=1, description="Test model")
    with pytest.raises(MlflowException, match=r"Model Version .+ not found"):
        client.transition_model_version_stage(name=name, version=1, stage="Staging")

    client.delete_model_version(name, 3)
    assert {mv.version for mv in client.search_model_versions("name = '%s'" % name)} == {"2"}

    # new model versions will not reuse existing version numbers
    mv4 = client.create_model_version(name, "runs:/run_id_2/a/b/c", "run_id_2")
    assert mv4.version == "4"
    assert mv4.name == name
    assert {mv.version for mv in client.search_model_versions("name = '%s'" % name)} == {
        "2",
        "4",
    }


def test_set_delete_model_version_tag_flow(client):
    name = "SetDeleteMVTagTest"
    client.create_registered_model(name)
    client.create_model_version(name, "runs:/run_id/model", "run_id_1")
    model_version_detailed = client.get_model_version(name, "1")
    assert model_version_detailed.tags == {}
    tags = {"key": "value", "numeric value": 12345}
    for key, value in tags.items():
        client.set_model_version_tag(name, "1", key, value)
    model_version_detailed = client.get_model_version(name, "1")
    assert model_version_detailed.tags == {"key": "value", "numeric value": "12345"}
    client.delete_model_version_tag(name, "1", "key")
    model_version_detailed = client.get_model_version(name, "1")
    assert model_version_detailed.tags == {"numeric value": "12345"}


def test_set_model_version_tag_with_empty_string_as_value(client):
    name = "SetMVTagEmptyValueTest"
    client.create_registered_model(name)
    client.create_model_version(name, "runs:/run_id/model", "run_id_1")
    client.set_model_version_tag(name, "1", "tag_key", "")
    assert {"tag_key": ""}.items() <= client.get_model_version(name, "1").tags.items()


def test_set_delete_registered_model_alias_and_get_model_version_by_alias_flow(client):
    name = "SetDeleteGetRMAliasTest"
    client.create_registered_model(name)
    client.create_model_version(name, "runs:/run_id/model", "run_id_1")
    model = client.get_registered_model(name)
    assert model.aliases == {}
    mv = client.get_model_version(name, "1")
    assert mv.aliases == []
    client.set_registered_model_alias(name, "test_alias", "1")
    model = client.get_registered_model(name)
    assert model.aliases == {"test_alias": "1"}
    mv = client.get_model_version(name, "1")
    assert mv.aliases == ["test_alias"]
    mv_alias = client.get_model_version_by_alias(name, "test_alias")
    assert mv == mv_alias
    client.delete_registered_model_alias(name, "test_alias")
    model = client.get_registered_model(name)
    assert model.aliases == {}
    mv = client.get_model_version(name, "1")
    assert mv.aliases == []
