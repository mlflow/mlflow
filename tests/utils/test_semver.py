import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.semver import normalize_semver


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.2.3", "1.2.3"),
        ("1", "1.0.0"),
        ("1.0", "1.0.0"),
        ("1.0.0-beta.11", "1.0.0-beta.11"),
        ("1.0.0+build.5", "1.0.0+build.5"),
        ("1.0.0-rc.1+build.5", "1.0.0-rc.1+build.5"),
        ("1.2-beta.1", "1.2.0-beta.1"),
        ("  1.2.3  ", "1.2.3"),
    ],
)
def test_normalize_semver_valid(raw, expected):
    assert normalize_semver(raw) == expected


def test_normalize_semver_is_idempotent():
    assert normalize_semver(normalize_semver("1.0")) == "1.0.0"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "abc", "1.2.3.4", "01.2.3", "1.2.3-", "1.2.3-beta..1", "v1.2.3", "1.2.x"],
)
def test_normalize_semver_invalid(raw):
    with pytest.raises(MlflowException, match="[Ss]em[Vv]er|version"):
        normalize_semver(raw)


def test_normalize_semver_non_string():
    with pytest.raises(MlflowException, match="version"):
        normalize_semver(None)
