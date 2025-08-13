from unittest import mock

import pytest

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.utils.models import _parse_model_uri, get_model_name_and_version
from mlflow.tracking._model_registry.client import ModelRegistryClient


@pytest.mark.parametrize(
    ("uri", "expected_name", "expected_version"),
    [
        ("models:/AdsModel1/0", "AdsModel1", "0"),
        ("models:/Ads Model 1/12345", "Ads Model 1", "12345"),
        ("models://////Ads Model 1/12345", "Ads Model 1", "12345"),  # many slashes
        ("models:/12345/67890", "12345", "67890"),
        ("models://profile@databricks/12345/67890", "12345", "67890"),
        ("models:/catalog.schema.model/0", "catalog.schema.model", "0"),  # UC Model format
    ],
)
def test_parse_models_uri_with_version(uri, expected_name, expected_version):
    (model_id, name, version, stage, alias) = _parse_model_uri(uri)
    assert model_id is None
    assert name == expected_name
    assert version == expected_version
    assert stage is None
    assert alias is None


@pytest.mark.parametrize(
    ("uri", "expected_name", "expected_stage"),
    [
        ("models:/AdsModel1/Production", "AdsModel1", "Production"),
        ("models:/AdsModel1/production", "AdsModel1", "production"),  # case insensitive
        ("models:/AdsModel1/pROduction", "AdsModel1", "pROduction"),  # case insensitive
        ("models:/Ads Model 1/None", "Ads Model 1", "None"),
        ("models://////Ads Model 1/Staging", "Ads Model 1", "Staging"),  # many slashes
        ("models://scope:key@databricks/Ads Model 1/None", "Ads Model 1", "None"),
        (
            "models:/Name/Stage@Alias",
            "Name",
            "Stage@Alias",
        ),  # technically allowed, but the backend would throw
        ("models:/Name@Alias/Stage", "Name@Alias", "Stage"),
    ],
)
def test_parse_models_uri_with_stage(uri, expected_name, expected_stage):
    (model_id, name, version, stage, alias) = _parse_model_uri(uri)
    assert model_id is None
    assert name == expected_name
    assert version is None
    assert stage == expected_stage
    assert alias is None


@pytest.mark.parametrize(
    ("uri", "expected_name"),
    [
        ("models:/AdsModel1/latest", "AdsModel1"),
        ("models:/AdsModel1/Latest", "AdsModel1"),  # case insensitive
        ("models:/AdsModel1/LATEST", "AdsModel1"),  # case insensitive
        ("models:/Ads Model 1/latest", "Ads Model 1"),
        ("models://////Ads Model 1/latest", "Ads Model 1"),  # many slashes
        ("models://scope:key@databricks/Ads Model 1/latest", "Ads Model 1"),
        ("models:/catalog.schema.model/latest", "catalog.schema.model"),  # UC Model format
    ],
)
def test_parse_models_uri_with_latest(uri, expected_name):
    (model_id, name, version, stage, alias) = _parse_model_uri(uri)
    assert model_id is None
    assert name == expected_name
    assert version is None
    assert stage is None
    assert alias is None


@pytest.mark.parametrize(
    ("uri", "expected_name", "expected_alias"),
    [
        ("models:/AdsModel1@Champion", "AdsModel1", "Champion"),
        ("models:/AdsModel1@champion", "AdsModel1", "champion"),  # case insensitive
        ("models:/AdsModel1@cHAmpion", "AdsModel1", "cHAmpion"),  # case insensitive
        ("models:/Ads Model 1@challenger", "Ads Model 1", "challenger"),
        ("models://scope:key/Ads Model 1@None", "Ads Model 1", "None"),
        ("models://////Ads Model 1@TestAlias", "Ads Model 1", "TestAlias"),  # many slashes
        ("models:/catalog.schema.model@None", "catalog.schema.model", "None"),  # UC Model format
        ("models:/A!&#$%;{}()[]CrazyName@Alias", "A!&#$%;{}()[]CrazyName", "Alias"),
        (
            "models:/NameWith@IntheMiddle@Alias",
            "NameWith@IntheMiddle",
            "Alias",
        ),  # check for model name with alias
    ],
)
def test_parse_models_uri_with_alias(uri, expected_name, expected_alias):
    (model_id, name, version, stage, alias) = _parse_model_uri(uri)
    assert model_id is None
    assert name == expected_name
    assert version is None
    assert stage is None
    assert alias == expected_alias


def test_parse_models_uri_model_id():
    (model_id, name, version, stage, alias) = _parse_model_uri("models:/12345")
    assert model_id == "12345"
    assert name is None
    assert version is None
    assert stage is None
    assert alias is None


@pytest.mark.parametrize(
    "uri",
    [
        "notmodels:/NameOfModel/12345",  # wrong scheme with version
        "notmodels:/NameOfModel/StageName",  # wrong scheme with stage
        "notmodels:/NameOfModel@alias",  # wrong scheme with alias
        "models:/",  # no model name
        "models:/ /Stage",  # empty name
        "models:/Name/",  # empty suffix
        "models:/Name@",  # empty alias
        "models:/Name/Stage/0",  # too many specifiers
        "models:Name/Stage",  # missing slash
    ],
)
def test_parse_models_uri_invalid_input(uri):
    with pytest.raises(MlflowException, match="Not a proper models"):
        _parse_model_uri(uri)


def test_improper_model_uri_msg_prompts():
    from mlflow.store.artifact.utils.models import _improper_model_uri_msg

    uri = "prompts:/baduri"
    msg = _improper_model_uri_msg(uri, scheme="prompts")
    assert "prompts:/" in msg, f"Missing 'prompts:/' in message: {msg}"
    assert "Prompts URIs" in msg, f"Missing 'Prompts URIs' in message: {msg}"
    assert "prompts:/name/suffix" in msg, f"Missing 'prompts:/name/suffix' in message: {msg}"


def test_get_model_name_and_version_with_version():
    with mock.patch.object(
        ModelRegistryClient, "get_latest_versions", return_value=[]
    ) as mlflow_client_mock:
        assert get_model_name_and_version(MlflowClient(), "models:/AdsModel1/123") == (
            "AdsModel1",
            "123",
        )
        mlflow_client_mock.assert_not_called()


def test_get_model_name_and_version_with_stage():
    with mock.patch.object(
        ModelRegistryClient,
        "get_latest_versions",
        return_value=[
            ModelVersion(
                name="mv1", version="10", creation_timestamp=123, current_stage="Production"
            ),
            ModelVersion(
                name="mv2", version="15", creation_timestamp=124, current_stage="Production"
            ),
        ],
    ) as mlflow_client_mock:
        assert get_model_name_and_version(MlflowClient(), "models:/AdsModel1/Production") == (
            "AdsModel1",
            "15",
        )
        mlflow_client_mock.assert_called_once_with("AdsModel1", ["Production"])


def test_get_model_name_and_version_with_latest():
    with mock.patch.object(
        ModelRegistryClient,
        "get_latest_versions",
        return_value=[
            ModelVersion(
                name="mv1", version="10", creation_timestamp=123, current_stage="Production"
            ),
            ModelVersion(name="mv3", version="20", creation_timestamp=125, current_stage="None"),
            ModelVersion(name="mv2", version="15", creation_timestamp=124, current_stage="Staging"),
        ],
    ) as mlflow_client_mock:
        assert get_model_name_and_version(MlflowClient(), "models:/AdsModel1/latest") == (
            "AdsModel1",
            "20",
        )
        mlflow_client_mock.assert_called_once_with("AdsModel1", None)
        # Check that "latest" is case insensitive.
        assert get_model_name_and_version(MlflowClient(), "models:/AdsModel1/lATest") == (
            "AdsModel1",
            "20",
        )


def test_get_model_name_and_version_with_alias():
    with mock.patch.object(
        ModelRegistryClient,
        "get_model_version_by_alias",
        return_value=ModelVersion(
            name="mv1", version="10", creation_timestamp=123, aliases=["Champion"]
        ),
    ) as mlflow_client_mock:
        assert get_model_name_and_version(MlflowClient(), "models:/AdsModel1@Champion") == (
            "AdsModel1",
            "10",
        )
        mlflow_client_mock.assert_called_once_with("AdsModel1", "Champion")


@pytest.mark.parametrize(
    ("uri", "expected_name", "expected_version", "expected_alias"),
    [
        ("prompts:/Prompt1/1", "Prompt1", "1", None),
        ("prompts:/Prompt-Name_123/42", "Prompt-Name_123", "42", None),
        ("prompts:/Prompt1@production", "Prompt1", None, "production"),
        ("prompts:/Prompt1@PRODUCTION", "Prompt1", None, "PRODUCTION"),
        ("prompts:/Prompt1@prod-1", "Prompt1", None, "prod-1"),
        ("prompts:/complex.prompt_name@alias-42", "complex.prompt_name", None, "alias-42"),
    ],
)
def test_parse_prompts_uri(uri, expected_name, expected_version, expected_alias):
    parsed = _parse_model_uri(uri, scheme="prompts")
    assert parsed.name == expected_name
    assert parsed.version == expected_version
    assert parsed.alias == expected_alias


@pytest.mark.parametrize(
    "uri",
    [
        "notprompts:/Prompt1/1",  # wrong scheme
        "prompts:/",  # no prompt name
        "prompts:/ /1",  # empty name
        "prompts:/Prompt1/",  # empty version
        "prompts:/Prompt1@",  # empty alias
        "prompts:/Prompt1/1/2",  # too many specifiers
        "prompts:Prompt1/1",  # missing slash
        "prompts:/Prompt1/latest",  # 'latest' not supported for prompts
        "prompts:/Prompt1/Production",  # stage not supported for prompts
    ],
)
def test_parse_prompts_uri_invalid_input(uri):
    with pytest.raises(MlflowException, match="Not a proper prompts"):
        _parse_model_uri(uri, scheme="prompts")


def test_improper_model_uri_msg_invalid_scheme():
    from mlflow.store.artifact.utils.models import _improper_model_uri_msg

    with pytest.raises(ValueError, match="Unsupported scheme"):
        _improper_model_uri_msg("foo:/bar", scheme="foo")
