import warnings
from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest

from mlflow.mismatch import _check_version_mismatch


@pytest.mark.parametrize(
    ("mlflow_version", "skinny_version"),
    [
        ("1.0.0", "1.0.0"),
        ("1.0.0.dev0", "1.0.0"),
        ("1.0.0", "1.0.0.dev0"),
        ("1.0.0.dev0", "1.0.0.dev0"),
        ("1.0.0", None),
        (None, "1.0.0"),
        (None, None),
    ],
)
def test_check_version_mismatch_no_warn(mlflow_version: str, skinny_version: str):
    def mock_version(package_name: str) -> str:
        if package_name == "mlflow":
            if mlflow_version is None:
                raise PackageNotFoundError
            return mlflow_version
        elif package_name == "mlflow-skinny":
            if skinny_version is None:
                raise PackageNotFoundError
            return skinny_version
        raise ValueError(f"Unexpected package: {package_name}")

    with mock.patch("importlib.metadata.version", side_effect=mock_version) as mv:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_version_mismatch()

        mv.assert_called()


@pytest.mark.parametrize(
    ("mlflow_version", "skinny_version"),
    [
        ("1.0.0", "1.0.1"),
        ("1.0.0rc0", "1.0.0"),
    ],
)
def test_check_version_mismatch_warn(mlflow_version: str, skinny_version: str):
    def mock_version(package_name: str) -> str:
        if package_name == "mlflow":
            return mlflow_version
        elif package_name == "mlflow-skinny":
            return skinny_version
        raise ValueError(f"Unexpected package: {package_name}")

    with mock.patch("importlib.metadata.version", side_effect=mock_version) as mv:
        with pytest.warns(
            UserWarning,
            match=r"Versions of mlflow \([.\w]+\) and mlflow-skinny \([.\w]+\) are different",
        ):
            _check_version_mismatch()

        mv.assert_called()
