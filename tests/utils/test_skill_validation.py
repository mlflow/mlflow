import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.validation import (
    _validate_agent_plugin_name,
    _validate_organization_name,
    _validate_skill_alias,
    _validate_skill_artifact_path,
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


@pytest.mark.parametrize("name", ["pr-workflow", "pr.workflow", "1.0.0", "acme"])
def test_valid_agent_plugin_names(name):
    _validate_agent_plugin_name(name)


def test_agent_plugin_name_rejects_reserved_chars():
    _validate_agent_plugin_name("pr-workflow")
    with pytest.raises(MlflowException, match="plugin name"):
        _validate_agent_plugin_name("pr/workflow")


@pytest.mark.parametrize(
    "name",
    ["PR-workflow", "pr_workflow", "pr-workflow-", "pr..workflow", "pr--workflow", "a" * 65],
)
def test_agent_plugin_name_rejects_values_outside_rfc_contract(name):
    with pytest.raises(MlflowException, match="plugin name"):
        _validate_agent_plugin_name(name)


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


@pytest.mark.parametrize("org", ["", "acme", "acme.com", "acme-labs"])
def test_valid_organization_names(org):
    _validate_organization_name(org)


@pytest.mark.parametrize("org", ["@acme", "a/b", "a@b"])
def test_invalid_organization_names(org):
    with pytest.raises(MlflowException, match="[Oo]rganization"):
        _validate_organization_name(org)


@pytest.mark.parametrize("version", [1, 2, 999])
def test_valid_skill_versions(version):
    _validate_skill_version(version)


@pytest.mark.parametrize("version", [0, -1, "1", 1.0, True, None])
def test_invalid_skill_versions(version):
    with pytest.raises(MlflowException, match="positive integer"):
        _validate_skill_version(version)


def test_skill_alias_reuses_model_alias_rules():
    _validate_skill_alias("production")
    with pytest.raises(MlflowException, match="[Ll]atest"):
        _validate_skill_alias("latest")


def test_skill_tag_reuses_tag_rules():
    _validate_skill_tag("team", "platform")
    with pytest.raises(MlflowException, match="Missing value for required parameter"):
        _validate_skill_tag(None, "v")


@pytest.mark.parametrize("path", ["code-review", "a/b/c", "dir/file.md"])
def test_valid_artifact_paths(path):
    _validate_skill_artifact_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/abs",
        "../escape",
        "a/../b",
        "a//b",
        "",
        "a/%2e%2e/b",
        "C:\\evil",
        "a/%2e/b",
        "a%2f%2fb",
    ],
)
def test_invalid_artifact_paths(path):
    with pytest.raises(MlflowException, match="path"):
        _validate_skill_artifact_path(path)
