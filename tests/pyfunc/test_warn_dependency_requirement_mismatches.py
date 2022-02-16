import os
from unittest import mock

import pytest
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

from mlflow.pyfunc import _warn_dependency_requirement_mismatches


@pytest.mark.large
def test_warn_dependency_requirement_mismatches(tmpdir):
    import cloudpickle
    import mlflow.utils.requirements_utils

    class AnyStringWith(str):
        def __eq__(self, other):
            return self in other

    model_path = tmpdir
    req_file_path = os.path.join(tmpdir, "requirements.txt")
    with open(req_file_path, "w") as fp:
        fp.write(f"cloudpickle=={cloudpickle.__version__}\nscikit-learn=={sklearn.__version__}\n")

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

        # Test case: multiple mismatched errors printed
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
            mock_warning.assert_called_once_with(AnyStringWith(
                "Detected one or more mismatches between the model's dependencies "
                "and the current Python environment"
            ))
            mock_warning.assert_called_once_with(AnyStringWith(
                "scikit-learn (current: 999.99.11, required: "
                f"scikit-learn=={sklearn.__version__}"
            ))
            mock_warning.assert_called_once_with(AnyStringWith(
                " - cloudpickle (current: 999.99.22, required: "
                f"cloudpickle=={cloudpickle.__version__})"
            ))

        mock_warning.reset_mock()

        with open(req_file_path, "w") as fp:
            fp.write("scikit-learn>=0.8,<=0.9")

        # Test case: multiple version specifiers requirement is satisfied
        with mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            gen_mock_get_installed_version_fn({"scikit-learn": "0.8.1"}),
        ):
            _warn_dependency_requirement_mismatches(model_path)
            mock_warning.assert_not_called()

        mock_warning.reset_mock()

        # Test case: multiple version specifiers requirement is not satisfied
        with mock.patch(
            "mlflow.utils.requirements_utils._get_installed_version",
            gen_mock_get_installed_version_fn({"scikit-learn": "0.7.1"}),
        ):
            _warn_dependency_requirement_mismatches(model_path)
            mock_warning.assert_called_once_with(AnyStringWith(
                "scikit-learn (current: 0.7.1, required: scikit-learn>=0.8,<=0.9"
            ))

        mock_warning.reset_mock()

        # Test case: some required package is uninstalled.
        with open(req_file_path, "w") as fp:
            fp.write("uninstalled-pkg==1.2.3")
        _warn_dependency_requirement_mismatches(model_path)
        mock_warning.assert_called_once_with(AnyStringWith(
            " - uninstalled-pkg (current: uninstalled, required: uninstalled-pkg==1.2.3)"
        ))

    # Test the case unexpected error happen.
    def bad_check_requirement_satisfied(x):
        raise RuntimeError("check_requirement_satisfied_fn_failed")

    with mock.patch(
        "mlflow.pyfunc._check_requirement_satisfied",
        bad_check_requirement_satisfied,
    ), mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
        _warn_dependency_requirement_mismatches(tmpdir)
        mock_warning.assert_called_once_with(AnyStringWith(
            "Encountered an unexpected error (check_requirement_satisfied_fn_failed) while "
            "detecting model dependency mismatches"
        ))
