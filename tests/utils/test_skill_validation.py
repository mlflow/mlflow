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


@pytest.mark.parametrize("name", ["code-review", "123", "a.b_c-d", "x"])
def test_valid_skill_names(name):
    _validate_skill_name(name)


@pytest.mark.parametrize("name", ["", "@acme", "a/b", "a@b", "a#b", "a?b"])
def test_invalid_skill_names(name):
    with pytest.raises(MlflowException, match="[Ss]kill name"):
        _validate_skill_name(name)


def test_skill_name_too_long():
    with pytest.raises(MlflowException, match="exceed|length|255"):
        _validate_skill_name("a" * 256)


def test_agent_plugin_name_rejects_reserved_chars():
    _validate_agent_plugin_name("pr-workflow")
    with pytest.raises(MlflowException, match="plugin name"):
        _validate_agent_plugin_name("pr/workflow")


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


@pytest.mark.parametrize("path", ["/abs", "../escape", "a/../b", "a//b", ""])
def test_invalid_artifact_paths(path):
    with pytest.raises(MlflowException, match="path"):
        _validate_skill_artifact_path(path)
