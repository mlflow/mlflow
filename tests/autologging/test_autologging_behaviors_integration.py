# pylint: disable=unused-argument

import importlib
import logging
import pytest
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from itertools import permutations
from unittest import mock

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import gorilla
from mlflow.utils.autologging_utils import (
    safe_patch,
    get_autologging_config,
    autologging_is_disabled,
)

from tests.autologging.fixtures import test_mode_off
from tests.autologging.fixtures import reset_stderr  # pylint: disable=unused-import


pytestmark = pytest.mark.large


AUTOLOGGING_INTEGRATIONS_TO_TEST = {
    mlflow.sklearn: "sklearn",
    mlflow.keras: "keras",
    mlflow.xgboost: "xgboost",
    mlflow.lightgbm: "lightgbm",
    mlflow.pytorch: "torch",
    mlflow.gluon: "mxnet.gluon",
    mlflow.fastai: "fastai",
    mlflow.statsmodels: "statsmodels",
    mlflow.spark: "pyspark",
    mlflow.pyspark.ml: "pyspark",
}


@pytest.fixture(autouse=True, scope="module")
def import_integration_libraries():
    for library_module in AUTOLOGGING_INTEGRATIONS_TO_TEST.values():
        importlib.import_module(library_module)


@pytest.fixture(autouse=True)
def disable_autologging_at_test_end():
    # The yeild statement is to insure that code below is executed as teardown code.
    # This will avoid bleeding of an active autologging session from test suite.
    yield
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        integration.autolog(disable=True)


@pytest.fixture()
def setup_sklearn_model():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression()

    return X, y, model


@pytest.mark.parametrize("integration", AUTOLOGGING_INTEGRATIONS_TO_TEST.keys())
def test_autologging_integrations_expose_configs_and_support_disablement(integration):
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        integration.autolog(disable=False)

    integration_name = integration.autolog.integration_name

    assert not autologging_is_disabled(integration_name)
    assert not get_autologging_config(integration_name, "disable", True)

    integration.autolog(disable=True)

    assert autologging_is_disabled(integration_name)
    assert get_autologging_config(integration_name, "disable", False)


@pytest.mark.parametrize("integration", AUTOLOGGING_INTEGRATIONS_TO_TEST.keys())
def test_autologging_integrations_use_safe_patch_for_monkey_patching(integration):
    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        with mock.patch(
            "mlflow.utils.gorilla.apply", wraps=gorilla.apply
        ) as gorilla_mock, mock.patch(
            integration.__name__ + ".safe_patch", wraps=safe_patch
        ) as safe_patch_mock:
            # In `mlflow.xgboost.autolog()`, we enable autologging for XGBoost sklearn
            # models using `mlflow.sklearn._autolog()`. So besides `safe_patch` calls in
            # `mlflow.xgboost.autolog()`, we need to count additional `safe_patch` calls
            # in sklearn autologging routine as well.
            if integration.__name__ == "mlflow.xgboost":
                with mock.patch(
                    "mlflow.sklearn.safe_patch", wraps=safe_patch
                ) as xgb_sklearn_safe_patch_mock:
                    integration.autolog(disable=False)
                    safe_patch_call_count = (
                        safe_patch_mock.call_count + xgb_sklearn_safe_patch_mock.call_count
                    )
            else:
                integration.autolog(disable=False)
                safe_patch_call_count = safe_patch_mock.call_count

            assert safe_patch_call_count > 0
            # `safe_patch` leverages `gorilla.apply` in its implementation. Accordingly, we expect
            # that the total number of `gorilla.apply` calls to be equivalent to the number of
            # `safe_patch` calls. This verifies that autologging integrations are leveraging
            # `safe_patch`, rather than calling `gorilla.apply` directly (which does not provide
            # exception safety properties)
            assert safe_patch_call_count == gorilla_mock.call_count


def test_autolog_respects_exclusive_flag(setup_sklearn_model):
    x, y, model = setup_sklearn_model

    mlflow.sklearn.autolog(exclusive=True)
    run = mlflow.start_run()
    model.fit(x, y)
    mlflow.end_run()
    run_data = MlflowClient().get_run(run.info.run_id).data
    metrics, params, tags = run_data.metrics, run_data.params, run_data.tags
    assert not metrics
    assert not params
    assert all("mlflow." in key for key in tags)

    mlflow.sklearn.autolog(exclusive=False)
    run = mlflow.start_run()
    model.fit(x, y)
    mlflow.end_run()
    run_data = MlflowClient().get_run(run.info.run_id).data
    metrics, params = run_data.metrics, run_data.params
    assert metrics
    assert params


def test_autolog_respects_disable_flag(setup_sklearn_model):
    x, y, model = setup_sklearn_model

    mlflow.sklearn.autolog(disable=True, exclusive=False)
    run = mlflow.start_run()
    model.fit(x, y)
    mlflow.end_run()
    run_data = MlflowClient().get_run(run.info.run_id).data
    metrics, params, tags = run_data.metrics, run_data.params, run_data.tags
    assert not metrics
    assert not params
    assert all("mlflow." in key for key in tags)

    mlflow.sklearn.autolog(disable=False, exclusive=False)
    run = mlflow.start_run()
    model.fit(x, y)
    mlflow.end_run()
    run_data = MlflowClient().get_run(run.info.run_id).data
    metrics, params = run_data.metrics, run_data.params
    assert metrics
    assert params


def test_autolog_reverts_patched_code_when_disabled():
    # use `KMeans` because it implements `fit`, `fit_transform`, and `fit_predict`.
    from sklearn.cluster import KMeans

    # Before any patching.
    model = KMeans()
    original_fit = model.fit
    original_fit_transform = model.fit_transform
    original_fit_predict = model.fit_predict

    # After patching.
    mlflow.sklearn.autolog(disable=False)
    patched_fit = model.fit
    patched_fit_transform = model.fit_transform
    patched_fit_predict = model.fit_predict
    assert patched_fit != original_fit
    assert patched_fit_transform != original_fit_transform
    assert patched_fit_predict != original_fit_predict

    # After revert of patching.
    mlflow.sklearn.autolog(disable=True)
    reverted_fit = model.fit
    reverted_fit_transform = model.fit_transform
    reverted_fit_predict = model.fit_predict

    assert reverted_fit == original_fit
    assert reverted_fit_transform == original_fit_transform
    assert reverted_fit_predict == original_fit_predict
    assert reverted_fit != patched_fit
    assert reverted_fit_transform != patched_fit_transform
    assert reverted_fit_predict != patched_fit_predict


def test_autolog_respects_disable_flag_across_import_orders():
    def test():
        from sklearn import svm, datasets

        iris = datasets.load_iris()
        svc = svm.SVC(C=2.0, degree=5, kernel="rbf")
        run = mlflow.start_run()
        svc.fit(iris.data, iris.target)
        mlflow.end_run()
        run_data = MlflowClient().get_run(run.info.run_id).data
        metrics, params, tags = run_data.metrics, run_data.params, run_data.tags
        assert not metrics
        assert not params
        assert all("mlflow." in key for key in tags)

    def import_sklearn():
        import sklearn  # pylint: disable=unused-import

    def disable_autolog():
        mlflow.sklearn.autolog(disable=True)

    def mlflow_autolog():
        mlflow.autolog()

    import_list = [import_sklearn, disable_autolog, mlflow_autolog]

    for func_order_list in permutations(import_list):
        for fun in func_order_list:
            fun()
        test()


@pytest.mark.usefixtures(test_mode_off.__name__)
def test_autolog_respects_silent_mode(tmpdir):
    # Use file-based experiment storage for this test. Otherwise, concurrent experiment creation in
    # multithreaded contexts may fail for other storage backends (e.g. SQLAlchemy)
    mlflow.set_tracking_uri(str(tmpdir))
    mlflow.set_experiment("test_experiment")

    og_showwarning = warnings.showwarning
    stream = StringIO()
    sys.stderr = stream
    logger = logging.getLogger(mlflow.__name__)

    from sklearn import datasets

    iris = datasets.load_iris()

    def train_model():
        import sklearn.utils
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV

        parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
        svc = svm.SVC()
        with sklearn.utils.parallel_backend(backend="threading"):
            clf = GridSearchCV(svc, parameters)
            clf.fit(iris.data, iris.target)

        return True

    # Call general and framework-specific autologging APIs to cover a
    # larger surface area for testing purposes
    mlflow.autolog(silent=True)
    mlflow.sklearn.autolog(silent=True, log_input_examples=True)

    executions = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for _ in range(2):
            e = executor.submit(train_model)
            executions.append(e)

    assert all([e.result() is True for e in executions])
    assert not stream.getvalue()
    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.getvalue()

    stream.truncate(0)

    mlflow.sklearn.autolog(silent=False, log_input_examples=True)

    executions = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        for _ in range(100):
            e = executor.submit(train_model)
            executions.append(e)

    assert all([e.result() is True for e in executions])
    assert stream.getvalue()
    # Verify that `warnings.showwarning` was restored to its original value after training
    # and that MLflow event logs are enabled
    assert warnings.showwarning == og_showwarning
    logger.info("verify that event logs are enabled")
    assert "verify that event logs are enabled" in stream.getvalue()

    # TODO: Investigate why this test occasionally leaks a run, which causes the
    # `clean_up_leaked_runs` fixture in `tests/conftest.py` to fail.
    while mlflow.active_run():
        mlflow.end_run()


def test_autolog_globally_configured_flag_set_correctly():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    AUTOLOGGING_INTEGRATIONS.clear()
    import sklearn  # pylint: disable=unused-import,unused-variable
    import pyspark  # pylint: disable=unused-import,unused-variable
    import pyspark.ml  # pylint: disable=unused-import,unused-variable

    integrations_to_test = ["sklearn", "spark", "pyspark.ml"]
    mlflow.autolog()
    for integration_name in integrations_to_test:
        assert AUTOLOGGING_INTEGRATIONS[integration_name]["globally_configured"]

    mlflow.sklearn.autolog()
    mlflow.spark.autolog()
    mlflow.pyspark.ml.autolog()

    for integration_name in integrations_to_test:
        assert "globally_configured" not in AUTOLOGGING_INTEGRATIONS[integration_name]
