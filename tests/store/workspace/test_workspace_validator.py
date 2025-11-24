import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.workspace.abstract_store import WorkspaceNameValidator


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("team-a", True),
        ("ab", True),
        ("a" * 63, True),
        ("t", False),
        ("a" * 64, False),
        ("Team-A", False),
        ("team_a", False),
        ("default", False),
        ("workspaces", False),
        ("-team", False),
        ("team-", False),
        ("a" * 256, False),
        ("123", True),
        ("a1-b2", True),
    ],
)
def test_workspace_name_validator_is_valid(name, expected):
    assert WorkspaceNameValidator.is_valid(name) is expected


@pytest.mark.parametrize(
    ("name", "error_fragment"),
    [
        (123, "must be a string"),
        ("t", "must be between"),
        ("Team-A", "must match the pattern"),
        ("default", "is reserved"),
    ],
)
def test_workspace_name_validator_validate_raises(name, error_fragment):
    with pytest.raises(MlflowException, match=error_fragment):
        WorkspaceNameValidator.validate(name)
