import os
import posixpath
import pytest
import requests


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


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # pylint: disable=unused-argument
    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()
    workflow_run_id = os.getenv("GITHUB_RUN_ID")
    github_token = os.getenv("GITHUB_TOKEN")
    github_job_name = os.getenv("GITHUB_JOB")
    if (
        "GITHUB_ACTIONS" not in os.environ
        or workflow_run_id is None
        or github_token is None
        or github_job_name is None
        or report.when in ("setup", "teardown")
        or report.outcome != "failed"
    ):
        return

    sess = requests.Session()
    sess.headers.update(
        {
            "Accept": "Accept: application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
        }
    )

    # Get the check run ID
    resp = sess.get(
        f"https://api.github.com/repos/mlflow/mlflow/actions/runs/{workflow_run_id}/jobs"
    )
    resp.raise_for_status()
    job_run_id = next(j["id"] for j in resp.json()["jobs"] if j["name"] == github_job_name)

    # Avoid adding too many annotations
    resp = sess.get(
        f"https://api.github.com/repos/mlflow/mlflow/check-runs/{job_run_id}/annotations"
    )
    resp.raise_for_status()
    annotations = resp.json()
    if len(annotations) > 10:
        return

    rel_file_path, lineno = report.location[:2]
    job_run_url = f"https://github.com/mlflow/mlflow/runs/{job_run_id}?check_suite_focus=true"
    print(job_run_url)
    lineno += 1
    annotation = {
        "path": rel_file_path,
        "start_line": lineno,
        "annotation_level": "failure",
        "title": "pytest failure",
        "message": f"{item.nodeid} failed in {job_run_url}.",
        "raw_details": "raw_details",
    }
    sess.post(
        f"https://api.github.com/repos/mlflow/mlflow/check-runs/{job_run_id}",
        json={
            "output": {
                "title": "test",
                "summary": "test",
                "annotations": [annotation],
            }
        },
    )
    resp.raise_for_status()
