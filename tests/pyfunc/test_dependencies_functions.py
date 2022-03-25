from unittest import mock
import pytest
import cloudpickle
import sklearn
import os

from mlflow.pyfunc import _warn_dependency_requirement_mismatches, get_model_dependencies
import mlflow.utils.requirements_utils

from tests.helper_functions import AnyStringWith


@pytest.mark.large
def test_warn_dependency_requirement_mismatches(tmpdir):
    req_file = tmpdir.join("requirements.txt")
    req_file.write(f"cloudpickle=={cloudpickle.__version__}\nscikit-learn=={sklearn.__version__}\n")

    with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
        # Test case: all packages satisfy requirements.
        _warn_dependency_requirement_mismatches(model_path=tmpdir)
        mock_warning.assert_not_called()

        mock_warning.reset_mock()

        original_get_installed_version_fn = mlflow.utils.requirements_utils._get_installed_version

        def gen_mock_get_installed_version_fn(mock_versions):
            def mock_get_installed_version_fn(package, module=None):
                if package in mock_versions:
                    return mock_versions[package]
                else:
                    return original_get_installed_version_fn(package, module)

            return mock_get_installed_version_fn

        # Test case: multiple mismatched packages
        with mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            gen_mock_get_installed_version_fn(
                {
                    "scikit-learn": "999.99.11",
                    "cloudpickle": "999.99.22",
                }
            ),
        ):
            _warn_dependency_requirement_mismatches(model_path=tmpdir)
            mock_warning.assert_called_once_with(
                """
Detected one or more mismatches between the model's dependencies and the current Python environment:
 - cloudpickle (current: 999.99.22, required: cloudpickle=={cloudpickle_version})
 - scikit-learn (current: 999.99.11, required: scikit-learn=={sklearn_version})
""".strip().format(
                    sklearn_version=sklearn.__version__, cloudpickle_version=cloudpickle.__version__
                )
            )

        mock_warning.reset_mock()

        req_file.write("scikit-learn>=0.8,<=0.9")

        # Test case: requirement with multiple version specifiers is satisfied
        with mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            gen_mock_get_installed_version_fn({"scikit-learn": "0.8.1"}),
        ):
            _warn_dependency_requirement_mismatches(model_path=tmpdir)
            mock_warning.assert_not_called()

        mock_warning.reset_mock()

        # Test case: requirement with multiple version specifiers is not satisfied
        with mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            gen_mock_get_installed_version_fn({"scikit-learn": "0.7.1"}),
        ):
            _warn_dependency_requirement_mismatches(model_path=tmpdir)
            mock_warning.assert_called_once_with(
                AnyStringWith(" - scikit-learn (current: 0.7.1, required: scikit-learn>=0.8,<=0.9)")
            )

        mock_warning.reset_mock()

        # Test case: required package is uninstalled.
        req_file.write("uninstalled-pkg==1.2.3")
        _warn_dependency_requirement_mismatches(model_path=tmpdir)
        mock_warning.assert_called_once_with(
            AnyStringWith(
                " - uninstalled-pkg (current: uninstalled, required: uninstalled-pkg==1.2.3)"
            )
        )

        mock_warning.reset_mock()

        # Test case: requirement without version specifiers
        req_file.write("mlflow")
        _warn_dependency_requirement_mismatches(model_path=tmpdir)
        mock_warning.assert_not_called()

        mock_warning.reset_mock()

        # Test case: an unexpected error happens while detecting mismatched packages.
        with mock.patch(
            "mlflow.pyfunc._check_requirement_satisfied",
            side_effect=RuntimeError("check_requirement_satisfied_fn_failed"),
        ):
            _warn_dependency_requirement_mismatches(model_path=tmpdir)
            mock_warning.assert_called_once_with(
                AnyStringWith(
                    "Encountered an unexpected error "
                    "(RuntimeError('check_requirement_satisfied_fn_failed')) while "
                    "detecting model dependency mismatches"
                )
            )


@pytest.mark.large
def test_get_model_dependencies(tmpdir):
    with open(os.path.join(tmpdir.strpath, "MLmodel"), "w") as f:
        f.write(
            """
artifact_path: model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.7.12
model_uuid: 722a374a432f48f09ee85da92df13bca
run_id: 765e66a5ba404650be51cb02cda66f35"""
        )

    req_file_path = os.path.join(tmpdir.strpath, "requirements.txt")
    with open(req_file_path, "w") as f:
        f.write(
            """
mlflow
cloudpickle==2.0.0
scikit-learn==1.0.2"""
        )

    conda_yml_path = os.path.join(tmpdir.strpath, "conda.yaml")
    with open(conda_yml_path, "w") as f:
        f.write(
            """
channels:
- conda-forge
dependencies:
- python=3.7.12
- pip=22.0.3
- scikit-learn=0.22.0
- tensorflow=2.0.0
- pip:
  - mlflow
  - cloudpickle==2.0.0
  - scikit-learn==1.0.1
name: mlflow-env"""
        )

    model_path = tmpdir.strpath

    # Test getting pip dependencies
    assert get_model_dependencies(model_path, format="pip") == req_file_path

    # Test getting pip dependencies will print instructions on databricks
    with mock.patch("mlflow.pyfunc._logger.info") as mock_log_info:
        get_model_dependencies(model_path, format="pip")
        mock_log_info.assert_not_called()

        mock_log_info.reset_mock()
        with mock.patch("mlflow.pyfunc.is_in_databricks_runtime", lambda: True):
            get_model_dependencies(model_path, format="pip")
            mock_log_info.assert_called_once_with(
                "To install these model dependencies in your Databricks notebook, run the "
                f"following command: '%pip install -r {req_file_path}'."
            )

    # Test getting conda environment
    assert get_model_dependencies(model_path, format="conda") == conda_yml_path

    os.remove(req_file_path)

    # Test getting pip requirement file failed and fallback to extract pip section from conda.yaml
    with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
        pip_file_path = get_model_dependencies(model_path, format="pip")
        with open(pip_file_path, "r") as f:
            result = f.read()

        assert result.strip() == "mlflow\ncloudpickle==2.0.0\nscikit-learn==1.0.1"
        mock_warning.assert_called_once_with(
            "The following conda dependencies are excluded: scikit-learn=0.22.0, tensorflow=2.0.0."
        )
