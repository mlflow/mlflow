import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.validation import (
    MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH,
    _validate_organization_name,
    _validate_skill_alias,
    _validate_skill_name,
    _validate_skill_tag,
    _validate_skill_version,
)


@pytest.mark.parametrize("name", ["code-review", "x", "123"])
def test_valid_skill_names(name):
    _validate_skill_name(name)


@pytest.mark.parametrize("name", ["", "@acme", "a/b", "a@b", "a#b", "a?b"])
def test_invalid_skill_names(name):
    with pytest.raises(MlflowException, match="[Ss]kill name"):
        _validate_skill_name(name)


@pytest.mark.parametrize(
    "name",
    ["Code-review", "code_review", "code.review", "code-review-", "code--review", "a" * 65],
)
def test_skill_name_rejects_values_outside_rfc_contract(name):
    with pytest.raises(MlflowException, match="[Ss]kill name"):
        _validate_skill_name(name)


@pytest.mark.parametrize(
    "organization",
    ["Acme.io", "acme_labs", "acme.io.", "acme..io", "acme--labs", "a" * 65],
)
def test_organization_rejects_values_outside_rfc_contract(organization):
    with pytest.raises(MlflowException, match="[Oo]rganization"):
        _validate_organization_name(organization)


@pytest.mark.parametrize("organization", ["", "acme", "acme.io", "acme-labs"])
def test_organization_accepts_rfc_examples(organization):
    _validate_organization_name(organization)


def test_organization_rejects_none():
    with pytest.raises(MlflowException, match="[Oo]rganization"):
        _validate_organization_name(None)


@pytest.mark.parametrize("org", ["", "acme", "acme.com", "acme-labs"])
def test_valid_organization_names(org):
    _validate_organization_name(org)


@pytest.mark.parametrize("org", ["@acme", "a/b", "a@b"])
def test_invalid_organization_names(org):
    with pytest.raises(MlflowException, match="[Oo]rganization"):
        _validate_organization_name(org)


@pytest.mark.parametrize("version", [1, 2, 999, 2_147_483_647])
def test_valid_skill_versions(version):
    _validate_skill_version(version)


@pytest.mark.parametrize("version", [0, -1, "1", 1.0, True, None, 2_147_483_648])
def test_invalid_skill_versions(version):
    with pytest.raises(MlflowException, match="positive integer"):
        _validate_skill_version(version)


def test_skill_alias_reuses_model_alias_rules():
    _validate_skill_alias("production")
    with pytest.raises(MlflowException, match="[Ll]atest"):
        _validate_skill_alias("latest")


def test_skill_tag_validates_key_and_value():
    _validate_skill_tag("team", "platform")
    with pytest.raises(MlflowException, match="Missing value for required parameter"):
        _validate_skill_tag(None, "v")
    with pytest.raises(MlflowException, match="Tag value cannot be None"):
        _validate_skill_tag("team", None)


def test_skill_tag_uses_registry_tag_value_limits():
    value = "x" * 9000
    assert _validate_skill_tag("team", value).value == value

    with pytest.raises(MlflowException, match="exceeds the maximum length"):
        _validate_skill_tag("team", "x" * (MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH + 1))
