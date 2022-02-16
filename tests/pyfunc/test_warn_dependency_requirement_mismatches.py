from unittest import mock
import pytest
import cloudpickle
import sklearn

from mlflow.pyfunc import _warn_dependency_requirement_mismatches
import mlflow.utils.requirements_utils

from tests.helper_functions import AnyStringWith


@pytest.mark.large
def test_warn_dependency_requirement_mismatches(tmpdir):
    model_path = tmpdir
    req_file = tmpdir.join("requirements.txt")
    req_file.write(f"cloudpickle=={cloudpickle.__version__}\nscikit-learn=={sklearn.__version__}\n")

    with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
        # Test case: all packages satisfy requirements.
        _warn_dependency_requirement_mismatches(model_path)
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
            _warn_dependency_requirement_mismatches(model_path)
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
            _warn_dependency_requirement_mismatches(model_path)
            mock_warning.assert_not_called()

        mock_warning.reset_mock()

        # Test case: requirement with multiple version specifiers is not satisfied
        with mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            gen_mock_get_installed_version_fn({"scikit-learn": "0.7.1"}),
        ):
            _warn_dependency_requirement_mismatches(model_path)
            mock_warning.assert_called_once_with(
                AnyStringWith(" - scikit-learn (current: 0.7.1, required: scikit-learn>=0.8,<=0.9)")
            )

        mock_warning.reset_mock()

        # Test case: required package is uninstalled.
        req_file.write("uninstalled-pkg==1.2.3")
        _warn_dependency_requirement_mismatches(model_path)
        mock_warning.assert_called_once_with(
            AnyStringWith(
                " - uninstalled-pkg (current: uninstalled, required: uninstalled-pkg==1.2.3)"
            )
        )

        mock_warning.reset_mock()

        # Test case: requirement without version specifiers
        req_file.write("mlflow")
        _warn_dependency_requirement_mismatches(model_path)
        mock_warning.assert_not_called()

        mock_warning.reset_mock()

        # Test case: an unexpected error happens while detecting mismatched packages.
        with mock.patch(
            "mlflow.pyfunc._check_requirement_satisfied",
            side_effect=RuntimeError("check_requirement_satisfied_fn_failed"),
        ):
            _warn_dependency_requirement_mismatches(tmpdir)
            mock_warning.assert_called_once_with(
                AnyStringWith(
                    "Encountered an unexpected error "
                    "(RuntimeError('check_requirement_satisfied_fn_failed')) while "
                    "detecting model dependency mismatches"
                )
            )
