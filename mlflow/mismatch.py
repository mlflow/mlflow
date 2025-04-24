import importlib.metadata
import warnings
from typing import Optional


def _get_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_version_mismatch() -> None:
    """
    Warns if both mlflow and mlflow-skinny are installed but their versions are different.

    Reference: https://github.com/pypa/pip/issues/4625
    """
    if (
        (mlflow_ver := _get_version("mlflow"))
        and ("dev" not in mlflow_ver)
        and (skinny_ver := _get_version("mlflow-skinny"))
        and ("dev" not in skinny_ver)
        and mlflow_ver != skinny_ver
    ):
        warnings.warn(
            (
                f"Versions of mlflow ({mlflow_ver}) and mlflow-skinny ({skinny_ver}) "
                "are different. This may lead to unexpected behavior. "
                "Please install the same version of both packages."
            ),
            stacklevel=2,
            category=UserWarning,
        )
