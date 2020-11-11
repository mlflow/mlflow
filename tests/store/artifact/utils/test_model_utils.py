import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.utils.model_utils import _parse_model_uri


@pytest.mark.parametrize(
    "uri, expected_name, expected_version",
    [
        ("models:/AdsModel1/0", "AdsModel1", 0),
        ("models:/Ads Model 1/12345", "Ads Model 1", 12345),
        ("models:/12345/67890", "12345", 67890),
        ("models://profile@databricks/12345/67890", "12345", 67890),
    ],
)
def test_parse_models_uri_with_version(uri, expected_name, expected_version):
    (name, version, stage) = _parse_model_uri(uri)
    assert name == expected_name
    assert version == expected_version
    assert stage is None


@pytest.mark.parametrize(
    "uri, expected_name, expected_stage",
    [
        ("models:/AdsModel1/Production", "AdsModel1", "Production"),
        ("models:/Ads Model 1/None", "Ads Model 1", "None"),
        ("models://scope:key@databricks/Ads Model 1/None", "Ads Model 1", "None"),
    ],
)
def test_parse_models_uri_with_stage(uri, expected_name, expected_stage):
    (name, version, stage) = _parse_model_uri(uri)
    assert name == expected_name
    assert version is None
    assert stage == expected_stage


@pytest.mark.parametrize(
    "uri",
    [
        "notmodels:/NameOfModel/12345",  # wrong scheme with version
        "notmodels:/NameOfModel/StageName",  # wrong scheme with stage
        "models:/",  # no model name
        "models:/Name/Stage/0",  # too many specifiers
        "models:Name/Stage",  # missing slash
        "models://Name/Stage",  # hostnames are ignored, path too short
    ],
)
def test_parse_models_uri_invalid_input(uri):
    with pytest.raises(MlflowException):
        _parse_model_uri(uri)
