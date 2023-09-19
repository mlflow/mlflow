import json
import os
import posixpath
import shutil
import subprocess

import click
import pytest

from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_TRACKING_URI


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


def pytest_configure(config):
    # Register markers to suppress `PytestUnknownMarkWarning`
    config.addinivalue_line("markers", "requires_ssh")
    config.addinivalue_line("markers", "notrackingurimock")
    config.addinivalue_line("markers", "allow_infer_pip_requirements_fallback")


def pytest_sessionstart(session):
    if uri := MLFLOW_TRACKING_URI.get():
        click.echo(
            click.style(
                (
                    f"Environment variable {MLFLOW_TRACKING_URI} is set to {uri!r}, "
                    "which may interfere with tests."
                ),
                fg="red",
            )
        )


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
            # Tests of flavor modules.
            "tests/azureml",
            "tests/catboost",
            "tests/diviner",
            "tests/fastai",
            "tests/gluon",
            "tests/h2o",
            "tests/johnsnowlabs",
            "tests/keras",
            "tests/keras_core",
            "tests/langchain",
            "tests/lightgbm",
            "tests/mleap",
            "tests/models",
            "tests/onnx",
            "tests/openai",
            "tests/paddle",
            "tests/pmdarima",
            "tests/prophet",
            "tests/pyfunc",
            "tests/pytorch",
            "tests/sagemaker",
            "tests/sentence_transformers",
            "tests/shap",
            "tests/sklearn",
            "tests/spacy",
            "tests/spark",
            "tests/statsmodels",
            "tests/tensorflow",
            "tests/transformers",
            "tests/xgboost",
            # Lazy loading test.
            "tests/test_mlflow_lazily_imports_ml_packages.py",
            # Tests of utils.
            "tests/utils/test_model_utils.py",
            # This test is included here because it imports many big libraries like tf, keras, etc.
            "tests/tracking/fluent/test_fluent_autolog.py",
            # Cross flavor autologging related tests.
            "tests/autologging/test_autologging_safety_unit.py",
            "tests/autologging/test_autologging_behaviors_unit.py",
            "tests/autologging/test_autologging_behaviors_integration.py",
            "tests/autologging/test_autologging_utils.py",
            "tests/autologging/test_training_session.py",
            # Opt in authentication feature.
            "tests/server/auth",
            "tests/gateway",
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


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(
    terminalreporter, exitstatus, config
):  # pylint: disable=unused-argument
    yield
    failed_test_reports = terminalreporter.stats.get("failed", [])
    if failed_test_reports:
        if len(failed_test_reports) <= 30:
            terminalreporter.section("command to run failed test cases")
            ids = [repr(report.nodeid) for report in failed_test_reports]
        else:
            terminalreporter.section("command to run failed test suites")
            # Use dict.fromkeys to preserve the order
            ids = list(dict.fromkeys(report.fspath for report in failed_test_reports))
        terminalreporter.write(" ".join(["pytest"] + ids))
        terminalreporter.write("\n" * 2)


@pytest.fixture(scope="module", autouse=True)
def clean_up_envs():
    yield

    if "GITHUB_ACTIONS" in os.environ:
        from mlflow.utils.virtualenv import _get_mlflow_virtualenv_root

        shutil.rmtree(_get_mlflow_virtualenv_root(), ignore_errors=True)
        if os.name != "nt":
            conda_info = json.loads(subprocess.check_output(["conda", "info", "--json"], text=True))
            root_prefix = conda_info["root_prefix"]
            for env in conda_info["envs"]:
                if env != root_prefix:
                    shutil.rmtree(env, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def enable_mlflow_testing():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_MLFLOW_TESTING.name, "TRUE")
        yield
