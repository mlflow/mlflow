import importlib.metadata
import warnings


def _get_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_version_mismatch() -> None:
    """
    Warns if both mlflow and child packages are installed but their versions are different.

    Reference: https://github.com/pypa/pip/issues/4625
    """
    mlflow_ver = _get_version("mlflow")
    # Skip if mlflow is installed from source.
    if mlflow_ver is None or "dev" in mlflow_ver:
        return

    child_packages = ["mlflow-skinny", "mlflow-tracing"]
    child_versions = [(p, _get_version(p)) for p in child_packages]

    mismatched = [
        (p, v) for p, v in child_versions if v is not None and "dev" not in v and v != mlflow_ver
    ]

    if mismatched:
        mismatched_str = ", ".join(f"{name} ({ver})" for name, ver in mismatched)
        warnings.warn(
            (
                f"Versions of mlflow ({mlflow_ver}) and child packages {mismatched_str} "
                "are different. This may lead to unexpected behavior. "
                "Please install the same version of all MLflow packages."
            ),
            stacklevel=2,
            category=UserWarning,
        )
