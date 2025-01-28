import importlib.metadata
import warnings
from typing import Optional

from packaging.version import Version


def _get_version(package_name: str) -> Optional[Version]:
    try:
        return Version(importlib.metadata.version(package_name))
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_version_mismatch() -> None:
    """
    Warns if both mlflow and mlflow-skinny are installed but their versions are different.

    Reference: https://github.com/pypa/pip/issues/4625
    """
    if (
        (mlflow_ver := _get_version("mlflow"))
        and (not mlflow_ver.is_devrelease)
        and (skinny_ver := _get_version("mlflow-skinny"))
        and (not skinny_ver.is_devrelease)
        and mlflow_ver != skinny_ver
    ):
        return warnings.warn(
            (
                f"Versions of mlflow ({mlflow_ver}) and mlflow-skinny ({skinny_ver}) "
                "are different. This may lead to unexpected behavior. "
                "Please install the same version of both packages."
            ),
            stacklevel=2,
            category=UserWarning,
        )
