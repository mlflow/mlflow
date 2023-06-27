"""
Simple unit tests to confirm that ModelRegistryClient properly calls the registry Store methods
and returns values when required.
"""

import pytest
from unittest import mock
from unittest.mock import ANY, patch

from mlflow.entities.model_registry import (
    ModelVersion,
    RegisteredModel,
    RegisteredModelTag,
    ModelVersionTag,
)
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
)
from mlflow.tracking._model_registry.client import ModelRegistryClient


@pytest.fixture
def mock_store():
    with mock.patch("mlflow.tracking._model_registry.utils._get_store") as mock_get_store:
        yield mock_get_store.return_value


def newModelRegistryClient():
    return ModelRegistryClient("uri:/fake", "uri:/fake")


def _model_version(
    name, version, stage, source="some:/source", run_id="run13579", tags=None, aliases=None
):
    return ModelVersion(
        name,
        version,
        "2345671890",
        "234567890",
        "some description",
        "UserID",
        stage,
        source,
        run_id,
        tags=tags,
        aliases=aliases,
    )


# Registered Model API
def test_create_registered_model(mock_store):
    tags_dict = {"key": "value", "another key": "some other value"}
    tags = [RegisteredModelTag(key, value) for key, value in tags_dict.items()]
    description = "such a great model"
    mock_store.create_registered_model.return_value = RegisteredModel(
        "Model 1", tags=tags, description=description
    )
    result = newModelRegistryClient().create_registered_model("Model 1", tags_dict, description)
    mock_store.create_registered_model.assert_called_once_with("Model 1", tags, description)
    assert result.name == "Model 1"
    assert result.tags == tags_dict


def test_update_registered_model(mock_store):
    name = "Model 1"
    new_description = "New Description"
    new_description_2 = "New Description 2"
    mock_store.update_registered_model.return_value = RegisteredModel(
        name, description=new_description
    )

    result = newModelRegistryClient().update_registered_model(
        name=name, description=new_description
    )
    mock_store.update_registered_model.assert_called_with(name=name, description=new_description)
    assert result.description == new_description

    mock_store.update_registered_model.return_value = RegisteredModel(
        name, description=new_description_2
    )
    result = newModelRegistryClient().update_registered_model(
        name=name, description=new_description_2
    )
    mock_store.update_registered_model.assert_called_with(
        name=name, description="New Description 2"
    )
    assert result.description == new_description_2


def test_rename_registered_model(mock_store):
    name = "Model 1"
    new_name = "New Name"
    mock_store.rename_registered_model.return_value = RegisteredModel(new_name)
    result = newModelRegistryClient().rename_registered_model(name=name, new_name=new_name)
    mock_store.rename_registered_model.assert_called_with(name=name, new_name=new_name)
    assert result.name == "New Name"

    mock_store.rename_registered_model.return_value = RegisteredModel("New Name 2")
    result = newModelRegistryClient().rename_registered_model(name=name, new_name="New Name 2")
    mock_store.rename_registered_model.assert_called_with(name=name, new_name="New Name 2")
    assert result.name == "New Name 2"


def test_update_registered_model_validation_errors_on_empty_new_name(mock_store):
    # pylint: disable=unused-argument
    with pytest.raises(MlflowException, match="The name must not be an empty string"):
        newModelRegistryClient().rename_registered_model("Model 1", " ")


def test_delete_registered_model(mock_store):
    newModelRegistryClient().delete_registered_model("Model 1")
    mock_store.delete_registered_model.assert_called_once()


def test_search_registered_models(mock_store):
    mock_store.search_registered_models.return_value = PagedList(
        [RegisteredModel("Model 1"), RegisteredModel("Model 2")], ""
    )
    result = newModelRegistryClient().search_registered_models(filter_string="test filter")
    mock_store.search_registered_models.assert_called_with(
        "test filter", SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT, None, None
    )
    assert len(result) == 2
    assert result.token == ""

    result = newModelRegistryClient().search_registered_models(
        filter_string="another filter",
        max_results=12,
        order_by=["A", "B DESC"],
        page_token="next one",
    )
    mock_store.search_registered_models.assert_called_with(
        "another filter", 12, ["A", "B DESC"], "next one"
    )
    assert len(result) == 2
    assert result.token == ""

    mock_store.search_registered_models.return_value = PagedList(
        [RegisteredModel("model A"), RegisteredModel("Model zz"), RegisteredModel("Model b")],
        "page 2 token",
    )
    result = newModelRegistryClient().search_registered_models(max_results=5)
    mock_store.search_registered_models.assert_called_with(None, 5, None, None)
    assert [rm.name for rm in result] == ["model A", "Model zz", "Model b"]
    assert result.token == "page 2 token"


def test_get_registered_model_details(mock_store):
    name = "Model 1"
    tags = [
        RegisteredModelTag("key", "value"),
        RegisteredModelTag("another key", "some other value"),
    ]
    mock_store.get_registered_model.return_value = RegisteredModel(
        name,
        "1263283747835",
        "1283168374623874",
        "I am a model",
        [
            _model_version("Model 1", 3, "None"),
            _model_version("Model 1", 2, "Staging"),
            _model_version("Model 1", 1, "Production"),
        ],
        tags=tags,
    )
    result = newModelRegistryClient().get_registered_model(name)
    mock_store.get_registered_model.assert_called_once()
    assert result.name == name
    assert len(result.latest_versions) == 3
    assert result.tags == {tag.key: tag.value for tag in tags}


def test_get_latest_versions(mock_store):
    mock_store.get_latest_versions.return_value = [
        _model_version("Model 1", 3, "None"),
        _model_version("Model 1", 2, "Staging"),
        _model_version("Model 1", 1, "Production"),
    ]
    result = newModelRegistryClient().get_latest_versions("Model 1", ["Stage1", "Stage2"])
    mock_store.get_latest_versions.assert_called_once_with(ANY, ["Stage1", "Stage2"])
    assert len(result) == 3


def test_set_registered_model_tag(mock_store):
    newModelRegistryClient().set_registered_model_tag("Model 1", "key", "value")
    mock_store.set_registered_model_tag.assert_called_once()


def test_delete_registered_model_tag(mock_store):
    newModelRegistryClient().delete_registered_model_tag("Model 1", "key")
    mock_store.delete_registered_model_tag.assert_called_once()


# Model Version API
@patch(
    "mlflow.tracking._model_registry.client.AWAIT_MODEL_VERSION_CREATE_SLEEP_DURATION_SECONDS", 1
)
def test_create_model_version_when_wait_exceeds_time(mock_store):
    name = "Model 1"
    version = "1"

    mv = ModelVersion(
        name=name, version=version, creation_timestamp=123, status="PENDING_REGISTRATION"
    )
    mock_store.create_model_version.return_value = mv
    mock_store.get_model_version.return_value = mv

    with pytest.raises(MlflowException, match="Exceeded max wait time"):
        newModelRegistryClient().create_model_version(
            name, "uri:/source", "run123", await_creation_for=1
        )


def test_create_model_version_does_not_wait_when_await_creation_param_is_false(mock_store):
    name = "Model 1"
    version = "1"

    mock_store.create_model_version.return_value = ModelVersion(
        name=name, version=version, creation_timestamp=123, status="PENDING_REGISTRATION"
    )

    result = newModelRegistryClient().create_model_version(
        name, "uri:/source", "run123", await_creation_for=None
    )
    result = newModelRegistryClient().create_model_version(
        name, "uri:/source", "run123", await_creation_for=0
    )

    mock_store.get_model_version.assert_not_called()

    assert result.name == name
    assert result.version == version


def test_create_model_version(mock_store):
    name = "Model 1"
    version = "1"
    tags_dict = {"key": "value", "another key": "some other value"}
    tags = [ModelVersionTag(key, value) for key, value in tags_dict.items()]
    description = "best model ever"

    mock_store.create_model_version.return_value = ModelVersion(
        name=name,
        version=version,
        creation_timestamp=123,
        tags=tags,
        run_link=None,
        description=description,
    )
    result = newModelRegistryClient().create_model_version(
        name, "uri:/for/source", "run123", tags_dict, None, description
    )
    mock_store.create_model_version.assert_called_once_with(
        name, "uri:/for/source", "run123", tags, None, description
    )

    assert result.name == name
    assert result.version == version
    assert result.tags == tags_dict


def test_create_model_version_no_run_id(mock_store):
    name = "Model 1"
    version = "1"
    tags_dict = {"key": "value", "another key": "some other value"}
    tags = [ModelVersionTag(key, value) for key, value in tags_dict.items()]
    description = "best model ever"

    mock_store.create_model_version.return_value = ModelVersion(
        name=name,
        version=version,
        creation_timestamp=123,
        tags=tags,
        run_link=None,
        description=description,
    )
    result = newModelRegistryClient().create_model_version(
        name, "uri:/for/source", tags=tags_dict, run_link=None, description=description
    )
    mock_store.create_model_version.assert_called_once_with(
        name, "uri:/for/source", None, tags, None, description
    )

    assert result.name == name
    assert result.version == version
    assert result.tags == tags_dict
    assert result.run_id is None


def test_update_model_version(mock_store):
    name = "Model 1"
    version = "12"
    description = "new description"
    expected_result = ModelVersion(name, version, creation_timestamp=123, description=description)
    mock_store.update_model_version.return_value = expected_result
    actal_result = newModelRegistryClient().update_model_version(name, version, "new description")
    mock_store.update_model_version.assert_called_once_with(
        name=name, version=version, description="new description"
    )
    assert expected_result == actal_result


def test_transition_model_version_stage(mock_store):
    name = "Model 1"
    version = "12"
    stage = "Production"
    expected_result = ModelVersion(name, version, creation_timestamp=123, current_stage=stage)
    mock_store.transition_model_version_stage.return_value = expected_result
    actual_result = newModelRegistryClient().transition_model_version_stage(name, version, stage)
    mock_store.transition_model_version_stage.assert_called_once_with(
        name=name, version=version, stage=stage, archive_existing_versions=False
    )
    assert expected_result == actual_result


def test_transition_model_version_stage_validation_errors(mock_store):
    # pylint: disable=unused-argument
    with pytest.raises(MlflowException, match="The stage must not be an empty string"):
        newModelRegistryClient().transition_model_version_stage("Model 1", "12", stage=" ")


def test_delete_model_version(mock_store):
    newModelRegistryClient().delete_model_version("Model 1", 12)
    mock_store.delete_model_version.assert_called_once()


def test_get_model_version_details(mock_store):
    tags = [ModelVersionTag("key", "value"), ModelVersionTag("another key", "some other value")]
    mock_store.get_model_version.return_value = _model_version(
        "Model 1", "12", "Production", tags=tags
    )
    result = newModelRegistryClient().get_model_version("Model 1", "12")
    mock_store.get_model_version.assert_called_once()
    assert result.name == "Model 1"
    assert result.tags == {tag.key: tag.value for tag in tags}


def test_get_model_version_download_uri(mock_store):
    mock_store.get_model_version_download_uri.return_value = "some:/uri/here"
    result = newModelRegistryClient().get_model_version_download_uri("Model 1", 12)
    mock_store.get_model_version_download_uri.assert_called_once()
    assert result == "some:/uri/here"


def test_search_model_versions(mock_store):
    mvs = [
        ModelVersion(
            name="Model 1", version="1", creation_timestamp=123, last_updated_timestamp=123
        ),
        ModelVersion(
            name="Model 1", version="2", creation_timestamp=124, last_updated_timestamp=124
        ),
        ModelVersion(
            name="Model 2", version="1", creation_timestamp=125, last_updated_timestamp=125
        ),
    ]
    mock_store.search_model_versions.return_value = PagedList(mvs[:2][::-1], "")
    result = newModelRegistryClient().search_model_versions("name=Model 1")
    mock_store.search_model_versions.assert_called_with(
        "name=Model 1", SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT, None, None
    )
    assert result == mvs[:2][::-1]
    assert result.token == ""

    mock_store.search_model_versions.return_value = PagedList([mvs[1], mvs[2], mvs[0]], "")
    result = newModelRegistryClient().search_model_versions(
        "version <= 2", max_results=2, order_by="version DESC", page_token="next"
    )
    mock_store.search_model_versions.assert_called_with("version <= 2", 2, "version DESC", "next")
    assert result == [mvs[1], mvs[2], mvs[0]]
    assert result.token == ""


def test_get_model_version_stages(mock_store):
    mock_store.get_model_version_stages.return_value = ["Stage A", "Stage B"]
    result = newModelRegistryClient().get_model_version_stages("Model 1", 1)
    mock_store.get_model_version_stages.assert_called_once()
    assert len(result) == 2
    assert "Stage A" in result
    assert "Stage B" in result


def test_set_model_version_tag(mock_store):
    newModelRegistryClient().set_model_version_tag("Model 1", "1", "key", "value")
    mock_store.set_model_version_tag.assert_called_once()


def test_delete_model_version_tag(mock_store):
    newModelRegistryClient().delete_model_version_tag("Model 1", "1", "key")
    mock_store.delete_model_version_tag.assert_called_once()


def test_set_registered_model_alias(mock_store):
    newModelRegistryClient().set_registered_model_alias("Model 1", "test_alias", "1")
    mock_store.set_registered_model_alias.assert_called_once()


def test_delete_registered_model_alias(mock_store):
    newModelRegistryClient().delete_registered_model_alias("Model 1", "test_alias")
    mock_store.delete_registered_model_alias.assert_called_once()


def test_get_model_version_by_alias(mock_store):
    mock_store.get_model_version_by_alias.return_value = _model_version(
        "Model 1", "12", "Production", aliases=["test_alias"]
    )
    result = newModelRegistryClient().get_model_version_by_alias("Model 1", "test_alias")
    mock_store.get_model_version_by_alias.assert_called_once()
    assert result.name == "Model 1"
    assert result.aliases == ["test_alias"]
