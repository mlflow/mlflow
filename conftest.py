import os
import posixpath
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--large-only",
        action="store_true",
        dest="large_only",
        default=False,
        help="Run only tests decorated with 'large' annotation",
    )
    parser.addoption(
        "--large",
        action="store_true",
        dest="large",
        default=False,
        help="Run tests decorated with 'large' annotation",
    )
    parser.addoption(
        "--requires-ssh",
        action="store_true",
        dest="requires_ssh",
        default=False,
        help="Run tests decorated with 'requires_ssh' annotation. "
        "These tests require keys to be configured locally "
        "for SSH authentication.",
    )
    parser.addoption(
        "--ignore-flavors",
        action="store_true",
        dest="ignore_flavors",
        default=False,
        help="Ignore tests for model flavors.",
    )
    parser.addoption(
        "--lazy-import",
        action="store_true",
        dest="lazy_import",
        default=False,
        help=(
            "Special flag that should be enabled when running "
            "tests/test_mlflow_lazily_imports_ml_packages.py"
        ),
    )


def pytest_configure(config):
    # Register markers to suppress `PytestUnknownMarkWarning`
    config.addinivalue_line("markers", "large: mark test as large")
    config.addinivalue_line("markers", "requires_ssh: mark test as requires_ssh")
    config.addinivalue_line("markers", "lazy_import: mark test as lazy_import")


def pytest_runtest_setup(item):
    marked_as_large = len([mark for mark in item.iter_markers(name="large")]) > 0
    if marked_as_large and not (
        item.config.getoption("--large") or item.config.getoption("--large-only")
    ):
        pytest.skip("use `--large` or `--large-only` to run this test")

    if not marked_as_large and item.config.getoption("--large-only"):
        pytest.skip("remove `--large-only` to run this test")

    marked_as_requires_ssh = len([mark for mark in item.iter_markers(name="requires_ssh")]) > 0
    if marked_as_requires_ssh and not item.config.getoption("--requires-ssh"):
        pytest.skip("use `--requires-ssh` to run this test")

    marked_as_lazy_import = len([mark for mark in item.iter_markers(name="lazy_import")]) > 0
    if marked_as_lazy_import and not item.config.getoption("--lazy-import"):
        pytest.skip("use `--lazy-import` to run this test")


@pytest.hookimpl(hookwrapper=True)
def pytest_ignore_collect(path, config):
    outcome = yield
    if not outcome.get_result() and config.getoption("ignore_flavors"):
        # If not ignored by the default hook and `--ignore-flavors` specified

        # Ignored files and directories must be included in dev/run-python-flavor-tests.sh
        model_flavors = [
            "tests/h2o",
            "tests/keras",
            "tests/pytorch",
            "tests/pyfunc",
            "tests/sagemaker",
            "tests/sklearn",
            "tests/spark",
            "tests/mleap",
            "tests/tensorflow",
            "tests/azureml",
            "tests/onnx",
            "tests/keras_autolog",
            "tests/tensorflow_autolog",
            "tests/gluon",
            "tests/gluon_autolog",
            "tests/xgboost",
            "tests/lightgbm",
            "tests/catboost",
            "tests/statsmodels",
            "tests/spacy",
            "tests/spark_autologging",
            "tests/fastai",
            "tests/models",
            "tests/shap",
            "tests/paddle",
            "tests/prophet",
            "tests/utils/test_model_utils.py",
            # this test is included here because it imports many big libraries like tf, keras, etc
            "tests/tracking/fluent/test_fluent_autolog.py",
            # cross flavor autologging related tests.
            "tests/autologging/test_autologging_safety_unit.py",
            "tests/autologging/test_autologging_behaviors_unit.py",
            "tests/autologging/test_autologging_behaviors_integration.py",
            "tests/autologging/test_autologging_utils.py",
            "tests/autologging/test_training_session.py",
        ]

        relpath = os.path.relpath(str(path))
        relpath = relpath.replace(os.sep, posixpath.sep)  # for Windows

        if relpath in model_flavors:
            outcome.force_result(True)
