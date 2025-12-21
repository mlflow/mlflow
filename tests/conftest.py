import inspect
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest
import requests
from opentelemetry import trace as trace_api

import mlflow
import mlflow.telemetry.utils
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_TRACKING_URI
from mlflow.telemetry.client import get_telemetry_client
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.export.inference_table import _TRACE_BUFFER
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.utils.os import is_windows
from mlflow.version import IS_TRACING_SDK_ONLY, VERSION

from tests.autologging.fixtures import enable_test_mode
from tests.helper_functions import get_safe_port
from tests.tracing.helper import purge_traces

if not IS_TRACING_SDK_ONLY:
    from mlflow.tracking._tracking_service.utils import _use_tracking_uri
    from mlflow.tracking.fluent import (
        _last_active_run_id,
        _reset_last_logged_model_id,
        clear_active_model,
    )


# Pytest hooks and configuration from root conftest.py
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
        "--splits",
        default=None,
        type=int,
        help="The number of groups to split tests into.",
    )
    parser.addoption(
        "--group",
        default=None,
        type=int,
        help="The group of tests to run.",
    )
    parser.addoption(
        "--serve-wheel",
        action="store_true",
        default=os.getenv("CI", "false").lower() == "true",
        help="Serve a wheel for the dev version of MLflow. True by default in CI, False otherwise.",
    )


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "requires_ssh")
    config.addinivalue_line("markers", "notrackingurimock")
    config.addinivalue_line("markers", "flaky: mark test as flaky to allow reruns")
    config.addinivalue_line("markers", "allow_infer_pip_requirements_fallback")
    config.addinivalue_line(
        "markers", "do_not_disable_new_import_hook_firing_if_module_already_exists"
    )
    config.addinivalue_line("markers", "classification")
    config.addinivalue_line("markers", "no_mock_requests_get")

    labels = fetch_pr_labels() or []
    if "fail-fast" in labels:
        config.option.maxfail = 1

    # Register SQLAlchemy LegacyAPIWarning filter only if sqlalchemy is available
    try:
        import sqlalchemy  # noqa: F401

        config.addinivalue_line("filterwarnings", "error::sqlalchemy.exc.LegacyAPIWarning")
    except ImportError:
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config):
    if not_exists := [p for p in config.getoption("ignore") or [] if not os.path.exists(p)]:
        raise pytest.UsageError(f"The following paths are ignored but do not exist: {not_exists}")

    group = config.getoption("group")
    splits = config.getoption("splits")

    if splits is None and group is None:
        return None

    if splits and group is None:
        raise pytest.UsageError("`--group` is required")

    if group and splits is None:
        raise pytest.UsageError("`--splits` is required")

    if splits < 0:
        raise pytest.UsageError("`--splits` must be >= 1")

    if group < 1 or group > splits:
        raise pytest.UsageError("`--group` must be between 1 and {splits}")

    return None


@dataclass
class TestResult:
    path: Path
    test_name: str
    execution_time: float


_test_results: list[TestResult] = []


def pytest_sessionstart(session):
    # Clear duration tracking state at the start of each session
    _test_results.clear()

    if IS_TRACING_SDK_ONLY:
        return

    import click

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


def to_md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]

    # Calculate column widths
    widths = [max(len(row[i]) for row in rows) for i in range(n)]

    def esc(s: str) -> str:
        return s.replace("|", r"\|").replace("\n", "<br>")

    # Format rows with proper padding
    def format_row(row: list[str]) -> str:
        cells = [esc(cell).ljust(width) for cell, width in zip(row, widths)]
        return "| " + " | ".join(cells) + " |"

    header = format_row(rows[0])
    sep = "| " + " | ".join(["-" * w for w in widths]) + " |"
    body = [format_row(row) for row in rows[1:]]

    return "\n".join([header, sep, *body])


def generate_duration_stats() -> str:
    """Generate per-file duration statistics as markdown table."""
    if not _test_results:
        return ""

    # Group results by file path
    file_groups: defaultdict[Path, list[float]] = defaultdict(list)
    for result in _test_results:
        file_groups[result.path].append(result.execution_time)

    rows = []
    for path, test_times in file_groups.items():
        rel_path = path.relative_to(Path.cwd()).as_posix()
        total_dur = sum(test_times)
        if total_dur < 1.0:
            # Ignore files with total duration < 1s
            continue
        test_count = len(test_times)
        min_test = min(test_times)
        max_test = max(test_times)
        avg_test = sum(test_times) / len(test_times)

        rows.append((rel_path, total_dur, test_count, min_test, max_test, avg_test))

    rows.sort(key=lambda r: r[1], reverse=True)

    if not rows:
        return ""

    # Prepare data for markdown table (headers + data rows)
    table_rows = [["Rank", "File", "Duration", "Tests", "Min", "Max", "Avg"]]
    for idx, (path, dur, count, min_, max_, avg_) in enumerate(rows, 1):
        table_rows.append(
            [
                str(idx),
                f"`{path}`",
                f"{dur:.2f}s",
                str(count),
                f"{min_:.3f}s",
                f"{max_:.3f}s",
                f"{avg_:.3f}s",
            ]
        )

    return to_md_table(table_rows)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """
    Custom test protocol that tracks test duration and supports rerunning failed tests
    marked with @pytest.mark.flaky.

    This is a simplified implementation inspired by pytest-rerunfailures:
    https://github.com/pytest-dev/pytest-rerunfailures/blob/365dc54ba3069f55a870cda2c3e1e3c33c68f326/src/pytest_rerunfailures.py#L564-L619

    Usage:
        @pytest.mark.flaky(attempts=3)
        def test_something():
            # Will run up to 3 times total if it keeps failing
            ...

        @pytest.mark.flaky(attempts=3, condition=sys.platform == "win32")
        def test_windows_only_flaky():
            ...
    """
    from _pytest.runner import runtestprotocol

    # Check if we should enable flaky rerun logic
    should_rerun = False
    attempts = 1
    if flaky_marker := item.get_closest_marker("flaky"):
        condition = flaky_marker.kwargs.get("condition", True)
        if condition:
            should_rerun = True
            attempts = flaky_marker.kwargs.get("attempts", 3)

    item.execution_count = 0
    need_to_run = True
    total_duration = 0.0

    while need_to_run:
        item.execution_count += 1
        start = time.perf_counter()
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
        total_duration += time.perf_counter() - start

        for report in reports:
            if should_rerun and report.when == "call" and report.failed:
                if item.execution_count < attempts:
                    report.outcome = "rerun"
                    # Re-initialize the test item for the next run
                    if hasattr(item, "_request"):
                        item._initrequest()
                    break
            item.ihook.pytest_runtest_logreport(report=report)
        else:
            # No rerun needed (passed or exhausted attempts), exit the loop
            need_to_run = False

    _test_results.append(
        TestResult(path=item.path, test_name=item.name, execution_time=total_duration)
    )
    return True  # Indicate that we handled this protocol


def pytest_runtest_setup(item):
    markers = [mark.name for mark in item.iter_markers()]
    if "requires_ssh" in markers and not item.config.getoption("--requires-ssh"):
        pytest.skip("use `--requires-ssh` to run this test")


def fetch_pr_labels():
    """
    Returns the labels associated with the current pull request.
    """
    if "GITHUB_ACTIONS" not in os.environ:
        return None

    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None

    with open(os.environ["GITHUB_EVENT_PATH"]) as f:
        pr_data = json.load(f)
        return [label["name"] for label in pr_data["pull_request"]["labels"]]


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    outcome = yield

    # Handle rerun outcome
    if report.outcome == "rerun":
        outcome.force_result(("rerun", "R", ("RERUN", {"yellow": True})))
        return

    if report.when == "call":
        try:
            import psutil
        except ImportError:
            return

        (*rest, result) = outcome.get_result()
        mem = psutil.virtual_memory()
        mem_used = mem.used / 1024**3
        mem_total = mem.total / 1024**3

        disk = psutil.disk_usage("/")
        disk_used = disk.used / 1024**3
        disk_total = disk.total / 1024**3
        outcome.force_result(
            (
                *rest,
                (
                    f"{result} | "
                    f"MEM {mem_used:.1f}/{mem_total:.1f} GB | "
                    f"DISK {disk_used:.1f}/{disk_total:.1f} GB"
                ),
            )
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_ignore_collect(collection_path, config):
    outcome = yield
    if not outcome.get_result() and config.getoption("ignore_flavors"):
        # If not ignored by the default hook and `--ignore-flavors` specified

        # Ignored files and directories must be included in dev/run-python-flavor-tests.sh
        model_flavors = [
            # Tests of flavor modules.
            "tests/ag2",
            "tests/agno",
            "tests/anthropic",
            "tests/autogen",
            "tests/azureml",
            "tests/bedrock",
            "tests/catboost",
            "tests/crewai",
            "tests/dspy",
            "tests/gemini",
            "tests/groq",
            "tests/h2o",
            "tests/johnsnowlabs",
            "tests/keras",
            "tests/keras_core",
            "tests/llama_index",
            "tests/langchain",
            "tests/langgraph",
            "tests/lightgbm",
            "tests/litellm",
            "tests/mistral",
            "tests/models",
            "tests/onnx",
            "tests/openai",
            "tests/paddle",
            "tests/pmdarima",
            "tests/prophet",
            "tests/pydantic_ai",
            "tests/pyfunc",
            "tests/pytorch",
            "tests/strands",
            "tests/haystack",
            "tests/semantic_kernel",
            "tests/sentence_transformers",
            "tests/shap",
            "tests/sklearn",
            "tests/smolagents",
            "tests/spacy",
            "tests/spark",
            "tests/statsmodels",
            "tests/tensorflow",
            "tests/transformers",
            "tests/xgboost",
            # Lazy loading test.
            "tests/test_mlflow_lazily_imports_ml_packages.py",
            # This test is included here because it imports many big libraries like tf, keras, etc.
            "tests/tracking/fluent/test_fluent_autolog.py",
            # Cross flavor autologging related tests.
            "tests/autologging/test_autologging_safety_unit.py",
            "tests/autologging/test_autologging_behaviors_unit.py",
            "tests/autologging/test_autologging_behaviors_integration.py",
            "tests/autologging/test_autologging_utils.py",
            "tests/autologging/test_training_session.py",
        ]

        relpath = os.path.relpath(str(collection_path))
        relpath = relpath.replace(os.sep, posixpath.sep)  # for Windows

        if relpath in model_flavors:
            outcome.force_result(True)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    # Executing `tests.server.test_prometheus_exporter` after `tests.server.test_handlers`
    # results in an error because Flask >= 2.2.0 doesn't allow calling setup method such as
    # `before_request` on the application after the first request. To avoid this issue,
    # execute `tests.server.test_prometheus_exporter` first by reordering the test items.
    items.sort(key=lambda item: item.module.__name__ != "tests.server.test_prometheus_exporter")

    # Select the tests to run based on the group and splits
    if (splits := config.getoption("--splits")) and (group := config.getoption("--group")):
        items[:] = items[(group - 1) :: splits]


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    yield

    # Display per-file durations
    if duration_stats := generate_duration_stats():
        terminalreporter.write("\n")
        header = "per-file durations (sorted)"
        terminalreporter.write_sep("=", header)
        terminalreporter.write(f"::group::{header}\n\n")
        terminalreporter.write(duration_stats)
        terminalreporter.write("\n\n::endgroup::\n")
        terminalreporter.write("\n")

    if (
        # `uv run` was used to run tests
        "UV" in os.environ
        # Tests failed because of missing dependencies
        and (errors := terminalreporter.stats.get("error"))
        and any(re.search(r"ModuleNotFoundError|ImportError", str(e.longrepr)) for e in errors)
    ):
        terminalreporter.write("\n")
        terminalreporter.section("HINTS", yellow=True)
        terminalreporter.write(
            "To run tests with additional packages, use:\n"
            "  uv run --with <package> pytest ...\n\n"
            "For multiple packages:\n"
            "  uv run --with <package1> --with <package2> pytest ...\n\n",
            yellow=True,
        )

    # If there are failed tests, display a command to run them
    if failed_test_reports := terminalreporter.stats.get("failed", []):
        if len(failed_test_reports) <= 30:
            ids = [repr(report.nodeid) for report in failed_test_reports]
        else:
            # Use dict.fromkeys to preserve the order
            ids = list(dict.fromkeys(report.fspath for report in failed_test_reports))
        terminalreporter.section("command to run failed tests")
        terminalreporter.write(" ".join(["pytest"] + ids))
        terminalreporter.write("\n" * 2)

        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            summary_path = Path(summary_path).resolve()
            with summary_path.open("a") as f:
                f.write("## Failed tests\n")
                f.write("Run the following command to run the failed tests:\n")
                f.write("```bash\n")
                f.write(" ".join(["pytest"] + ids) + "\n")
                f.write("```\n\n")

        # If some tests failed at installing mlflow, we suggest using `--serve-wheel` flag.
        # Some test cases try to install mlflow via pip e.g. model loading. They pins
        # mlflow version to install based on local environment i.e. dev version ahead of
        # the latest release, hence it's not found on PyPI. `--serve-wheel` flag was
        # introduced to resolve this issue, which starts local PyPI server and serve
        # an mlflow wheel based on local source code.
        # Ref: https://github.com/mlflow/mlflow/pull/10247
        msg = f"No matching distribution found for mlflow=={VERSION}"
        for rep in failed_test_reports:
            if any(msg in t for t in (rep.longreprtext, rep.capstdout, rep.capstderr)):
                terminalreporter.section("HINTS", yellow=True)
                terminalreporter.write(
                    f"Found test(s) that failed with {msg!r}. Adding"
                    " --serve-wheel` flag to your pytest command may help.\n\n",
                    yellow=True,
                )
                break

    main_thread = threading.main_thread()
    if threads := [t for t in threading.enumerate() if t is not main_thread]:
        terminalreporter.section("Remaining threads", yellow=True)
        for idx, thread in enumerate(threads, start=1):
            terminalreporter.write(f"{idx}: {thread}\n")

        # Uncomment this block to print tracebacks of non-daemon threads
        # if non_daemon_threads := [t for t in threads if not t.daemon]:
        #     frames = sys._current_frames()
        #     terminalreporter.section("Tracebacks of non-daemon threads", yellow=True)
        #     for thread in non_daemon_threads:
        #         thread.join(timeout=1)
        #         if thread.is_alive() and (frame := frames.get(thread.ident)):
        #             terminalreporter.section(repr(thread), sep="~")
        #             terminalreporter.write("".join(traceback.format_stack(frame)))

    try:
        import psutil
    except ImportError:
        pass
    else:
        current_process = psutil.Process()
        if children := current_process.children(recursive=True):
            terminalreporter.section("Remaining child processes", yellow=True)
            for idx, child in enumerate(children, start=1):
                terminalreporter.write(f"{idx}: {child}\n")


# Test fixtures from tests/conftest.py


@pytest.fixture(autouse=IS_TRACING_SDK_ONLY, scope="session")
def remote_backend_for_tracing_sdk_test():
    """
    A fixture to start a remote backend for testing mlflow-tracing package integration.
    Since the tracing SDK has to be tested in an environment that has minimal dependencies,
    we need to start a tracking backend in an isolated uv environment.
    """
    port = get_safe_port()
    # Start a remote backend to test mlflow-tracing package integration.
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow_root = os.path.dirname(os.path.dirname(__file__))
        with subprocess.Popen(
            [
                "uv",
                "run",
                "--directory",
                # Install from the dev version
                mlflow_root,
                "mlflow",
                "server",
                "--port",
                str(port),
            ],
            cwd=temp_dir,
        ) as process:
            print("Starting mlflow server on port 5000")  # noqa: T201
            try:
                for _ in range(60):
                    try:
                        response = requests.get(f"http://localhost:{port}")
                        if response.ok:
                            break
                    except requests.ConnectionError:
                        print("MLflow server is not responding yet.")  # noqa: T201
                        time.sleep(1)
                else:
                    raise RuntimeError("Failed to start server")

                mlflow.set_tracking_uri(f"http://localhost:{port}")

                yield

            finally:
                process.terminate()


@pytest.fixture(autouse=IS_TRACING_SDK_ONLY)
def tmp_experiment_for_tracing_sdk_test(monkeypatch):
    # Generate a random experiment name
    experiment_name = f"trace-unit-test-{uuid.uuid4().hex}"
    experiment = mlflow.set_experiment(experiment_name)

    # Reduce retries for speed up tests
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    yield

    purge_traces(experiment_id=experiment.experiment_id)


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def tracking_uri_mock(db_uri: str, request: pytest.FixtureRequest) -> Iterator[str | None]:
    if "notrackingurimock" not in request.keywords:
        with _use_tracking_uri(db_uri):
            yield db_uri
    else:
        yield None


@pytest.fixture(autouse=True)
def reset_active_experiment_id():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)


@pytest.fixture(autouse=True)
def reset_mlflow_uri():
    yield
    # Resetting these environment variables cause sqlalchemy store tests to run with a sqlite
    # database instead of mysql/postgresql/mssql.
    if "DISABLE_RESET_MLFLOW_URI_FIXTURE" not in os.environ:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ.pop("MLFLOW_REGISTRY_URI", None)
        try:
            from mlflow.tracking import set_registry_uri

            # clean up the registry URI to avoid side effects
            set_registry_uri(None)
        except ImportError:
            # tracing sdk does not have the registry module
            pass


@pytest.fixture(autouse=True)
def reset_tracing():
    """
    Reset the global state of the tracing feature.

    This fixture is auto-applied for cleaning up the global state between tests
    to avoid side effects.
    """
    yield

    # Reset OpenTelemetry and MLflow tracer setup
    mlflow.tracing.reset()

    # Clear other global state and singletons
    _set_last_active_trace_id(None)
    _TRACE_BUFFER.clear()
    InMemoryTraceManager.reset()
    IPythonTraceDisplayHandler._instance = None

    # Reset opentelemetry tracer provider as well
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False
    trace_api._TRACER_PROVIDER = None


def _is_span_active():
    span = trace_api.get_current_span()
    return (span is not None) and not isinstance(span, trace_api.NonRecordingSpan)


@pytest.fixture(autouse=True)
def validate_trace_finish():
    """
    Validate all spans are finished and detached from the context by the end of the each test.

    Leaked span is critical problem and also hard to find without an explicit check.
    """
    # When the span is leaked, it causes confusing test failure in the subsequent tests. To avoid
    # this and make the test failure more clear, we fail first here.
    if _is_span_active():
        pytest.skip(reason="A leaked active span is found before starting the test.")

    yield

    assert not _is_span_active(), (
        "A span is still active at the end of the test. All spans must be finished "
        "and detached from the context before the test ends. The leaked span context "
        "may cause other subsequent tests to fail."
    )


@pytest.fixture(autouse=True, scope="session")
def enable_test_mode_by_default_for_autologging_integrations():
    """
    Run all MLflow tests in autologging test mode, ensuring that errors in autologging patch code
    are raised and detected. For more information about autologging test mode, see the docstring
    for :py:func:`mlflow.utils.autologging_utils._is_testing()`.
    """
    yield from enable_test_mode()


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def clean_up_leaked_runs():
    """
    Certain test cases validate safety API behavior when runs are leaked. Leaked runs that
    are not cleaned up between test cases may result in cascading failures that are hard to
    debug. Accordingly, this fixture attempts to end any active runs it encounters and
    throws an exception (which reported as an additional error in the pytest execution output).
    """
    try:
        yield
        assert not mlflow.active_run(), (
            "test case unexpectedly leaked a run. Run info: {}. Run data: {}".format(
                mlflow.active_run().info, mlflow.active_run().data
            )
        )
    finally:
        while mlflow.active_run():
            mlflow.end_run()


def _called_in_save_model():
    for frame in inspect.stack()[::-1]:
        if frame.function == "save_model":
            return True
    return False


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def prevent_infer_pip_requirements_fallback(request):
    """
    Prevents `mlflow.models.infer_pip_requirements` from falling back in `mlflow.*.save_model`
    unless explicitly disabled via `pytest.mark.allow_infer_pip_requirements_fallback`.
    """
    from mlflow.utils.environment import _INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE

    def new_exception(msg, *_, **__):
        if msg == _INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE and _called_in_save_model():
            raise Exception(
                "`mlflow.models.infer_pip_requirements` should not fall back in"
                "`mlflow.*.save_model` during test"
            )

    if "allow_infer_pip_requirements_fallback" not in request.keywords:
        with mock.patch("mlflow.utils.environment._logger.exception", new=new_exception):
            yield
    else:
        yield


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def clean_up_mlruns_directory(request):
    """
    Clean up an `mlruns` directory on each test module teardown on CI to save the disk space.
    """
    yield

    # Only run this fixture on CI.
    if "GITHUB_ACTIONS" not in os.environ:
        return

    mlruns_dir = os.path.join(request.config.rootpath, "mlruns")
    if os.path.exists(mlruns_dir):
        try:
            shutil.rmtree(mlruns_dir)
        except OSError:
            if is_windows():
                raise
            # `shutil.rmtree` can't remove files owned by root in a docker container.
            subprocess.check_call(["sudo", "rm", "-rf", mlruns_dir])


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def clean_up_last_logged_model_id():
    """
    Clean up the last logged model ID stored in a thread local var.
    """
    _reset_last_logged_model_id()


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def clean_up_last_active_run():
    _last_active_run_id.set(None)


@pytest.fixture(scope="module", autouse=not IS_TRACING_SDK_ONLY)
def clean_up_envs():
    """
    Clean up virtualenvs and conda environments created during tests to save disk space.
    """
    yield

    if "GITHUB_ACTIONS" in os.environ:
        from mlflow.utils.virtualenv import _get_mlflow_virtualenv_root

        shutil.rmtree(_get_mlflow_virtualenv_root(), ignore_errors=True)
        if not is_windows():
            conda_info = json.loads(subprocess.check_output(["conda", "info", "--json"], text=True))
            root_prefix = conda_info["root_prefix"]
            regex = re.compile(r"mlflow-\w{32,}")
            for env in conda_info["envs"]:
                if env == root_prefix:
                    continue
                if regex.fullmatch(os.path.basename(env)):
                    shutil.rmtree(env, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def enable_mlflow_testing():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(_MLFLOW_TESTING.name, "TRUE")
        yield


@pytest.fixture(scope="session", autouse=not IS_TRACING_SDK_ONLY)
def serve_wheel(request, tmp_path_factory):
    """
    Models logged during tests have a dependency on the dev version of MLflow built from
    source (e.g., mlflow==1.20.0.dev0) and cannot be served because the dev version is not
    available on PyPI. This fixture serves a wheel for the dev version from a temporary
    PyPI repository running on localhost and appends the repository URL to the
    `PIP_EXTRA_INDEX_URL` environment variable to make the wheel available to pip.
    """
    from tests.helper_functions import get_safe_port

    if "COPILOT_AGENT_ACTION" in os.environ:
        yield  # pytest expects a generator fixture to yield
        return

    if not request.config.getoption("--serve-wheel"):
        yield  # pytest expects a generator fixture to yield
        return

    root = tmp_path_factory.mktemp("root")
    mlflow_dir = root.joinpath("mlflow")
    mlflow_dir.mkdir()
    port = get_safe_port()
    try:
        repo_root = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--show-toplevel",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        # Some tests run in a Docker container where git is not installed.
        # In this case, assume we're in the root of the repo.
        repo_root = "."

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--wheel-dir",
            mlflow_dir,
            "--no-deps",
            repo_root,
        ],
    )
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
        ],
        cwd=root,
    ) as prc:
        try:
            url = f"http://localhost:{port}"
            if existing_url := os.environ.get("PIP_EXTRA_INDEX_URL"):
                url = f"{existing_url} {url}"
            os.environ["PIP_EXTRA_INDEX_URL"] = url
            # Set the `UV_INDEX` environment variable to allow fetching the wheel from the
            # url when using `uv` as environment manager
            os.environ["UV_INDEX"] = f"mlflow={url}"
            yield
        finally:
            prc.terminate()


@pytest.fixture
def mock_s3_bucket():
    """
    Creates a mock S3 bucket using moto

    Returns:
        The name of the mock bucket.
    """
    import boto3
    import moto

    with moto.mock_s3():
        bucket_name = "mock-bucket"
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.fixture
def tmp_sqlite_uri(tmp_path):
    path = tmp_path.joinpath("mlflow.db").as_uri()
    return ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]


@pytest.fixture
def mock_databricks_serving_with_tracing_env(monkeypatch):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", "true")


@pytest.fixture(params=[True, False])
def mock_is_in_databricks(request):
    with mock.patch(
        "mlflow.models.model.is_in_databricks_runtime", return_value=request.param
    ) as mock_databricks:
        yield mock_databricks


@pytest.fixture(autouse=not IS_TRACING_SDK_ONLY)
def reset_active_model_context():
    yield
    clear_active_model()


@pytest.fixture(autouse=True)
def clean_up_telemetry_threads():
    yield
    if client := get_telemetry_client():
        client._clean_up()


@pytest.fixture(scope="session")
def cached_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates and caches a SQLite database to avoid repeated migrations for each test run.

    This is a session-scoped fixture that creates the database once per test session.
    Individual tests should copy this database to their own tmp_path to avoid conflicts.
    """
    tmp_dir = tmp_path_factory.mktemp("sqlite_db")
    db_path = tmp_dir / "mlflow.db"

    if IS_TRACING_SDK_ONLY:
        return db_path

    try:
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    except ImportError:
        return db_path

    db_uri = f"sqlite:///{db_path}"
    artifact_uri = (tmp_dir / "artifacts").as_uri()
    store = SqlAlchemyStore(db_uri, artifact_uri)
    store.engine.dispose()

    return db_path


@pytest.fixture
def db_uri(cached_db: Path) -> Iterator[str]:
    """Returns a fresh SQLite URI for each test by copying the cached database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "mlflow.db"

        if not IS_TRACING_SDK_ONLY and cached_db.exists():
            shutil.copy2(cached_db, db_path)

        yield f"sqlite:///{db_path}"


@pytest.fixture(autouse=True)
def clear_engine_map():
    """
    Clear the SQLAlchemy engine cache in all stores between tests.

    Each SQLAlchemy store caches engines by database URI to prevent connection pool leaks.
    This fixture clears the cache between tests to ensure test isolation and prevent
    engines from one test affecting another.
    """
    try:
        from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
        from mlflow.store.model_registry.sqlalchemy_store import (
            SqlAlchemyStore as ModelRegistrySqlAlchemyStore,
        )
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

        SqlAlchemyStore._engine_map.clear()
        ModelRegistrySqlAlchemyStore._engine_map.clear()
        SqlAlchemyJobStore._engine_map.clear()
    except ImportError:
        pass
