import os
import posixpath
from packaging.version import Version

import pytest

import mlflow


def pytest_addoption(parser):
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
        "--mlp-next-release",
        action="store_true",
        dest="run_mlp_next_release_tests",
        default=False,
        help=(
            "Run MLflow Pipelines tests that depend on the next MLflow Pipelines"
            " template(s) release"
        ),
    )


def pytest_configure(config):
    # Register markers to suppress `PytestUnknownMarkWarning`
    config.addinivalue_line("markers", "requires_ssh")
    config.addinivalue_line("markers", "notrackingurimock")
    config.addinivalue_line("markers", "allow_infer_pip_requirements_fallback")
    config.addinivalue_line("markers", "mlp_next_release")


def pytest_runtest_setup(item):
    markers = [mark.name for mark in item.iter_markers()]
    if "requires_ssh" in markers and not item.config.getoption("--requires-ssh"):
        pytest.skip("use `--requires-ssh` to run this test")


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
            "tests/gluon",
            "tests/xgboost",
            "tests/lightgbm",
            "tests/catboost",
            "tests/statsmodels",
            "tests/spacy",
            "tests/fastai",
            "tests/models",
            "tests/shap",
            "tests/paddle",
            "tests/prophet",
            "tests/pmdarima",
            "tests/diviner",
            "tests/test_mlflow_lazily_imports_ml_packages.py",
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


def pytest_collection_modifyitems(session, config, items):  # pylint: disable=unused-argument
    # Executing `tests.server.test_prometheus_exporter` after `tests.server.test_handlers`
    # results in an error because Flask >= 2.2.0 doesn't allow calling setup method such as
    # `before_request` on the application after the first request. To avoid this issue,
    # execute `tests.server.test_prometheus_exporter` first by reordering the test items.
    items.sort(key=lambda item: item.module.__name__ != "tests.server.test_prometheus_exporter")

    if config.getoption("run_mlp_next_release_tests"):
        skip_not_mlp_next_release = pytest.mark.skip(
            reason=(
                f"Only running MLflow Pipelines next release tests beyond MLflow version"
                f" {mlflow.__version__} because --mlp_next_release is specified"
            )
        )
        for item in items:
            mark = item.get_closest_marker("mlp_next_release")
            if not mark or (
                mark and mark.args and Version(mark.args[0]) <= Version(mlflow.__version__)
            ):
                item.add_marker(skip_not_mlp_next_release)
    else:
        skip_mlp_next_release = pytest.mark.skip(
            reason=(
                f"MLflow Pipelines next release tests beyond MLflow version {mlflow.__version__}"
                f" are disabled because --mlp_next_release isn't specified"
            ),
        )
        for item in items:
            mark = item.get_closest_marker("mlp_next_release")
            if mark and (
                not mark.args or (mark.args and Version(mark.args[0]) > Version(mlflow.__version__))
            ):
                item.add_marker(skip_mlp_next_release)
