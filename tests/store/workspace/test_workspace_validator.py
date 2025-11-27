import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.workspace.abstract_store import WorkspaceNameValidator


@pytest.mark.parametrize(
    "name",
    [
        "team-a",
        "ab",
        "a" * 63,
        "123",
        "a1-b2",
    ],
)
def test_workspace_name_validator_accepts_valid_names(name):
    WorkspaceNameValidator.validate(name)


@pytest.mark.parametrize(
    ("name", "error_fragment"),
    [
        (123, "must be a string"),
        ("t", "must be between"),
        ("a" * 64, "must be between"),
        ("a" * 256, "must be between"),
        ("Team-A", "must match the pattern"),
        ("team_a", "must match the pattern"),
        ("-team", "must match the pattern"),
        ("team-", "must match the pattern"),
        ("default", "is reserved"),
        ("workspaces", "is reserved"),
    ],
)
def test_workspace_name_validator_validate_raises(name, error_fragment):
    with pytest.raises(MlflowException, match=error_fragment):
        WorkspaceNameValidator.validate(name)
