import functools
import inspect
from mock import mock
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.datasets
import sklearn.model_selection
from scipy.stats import uniform

from mlflow.models import Model
from mlflow.models.signature import infer_signature
from mlflow.models.utils import _read_example
import mlflow.sklearn
from mlflow.entities import RunStatus
from mlflow.sklearn.utils import (
    _is_supported_version,
    _get_arg_names,
    _truncate_dict,
)
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.autologging_utils import try_mlflow_log
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)

FIT_FUNC_NAMES = ["fit", "fit_transform", "fit_predict"]
TRAINING_SCORE = "training_score"
ESTIMATOR_CLASS = "estimator_class"
ESTIMATOR_NAME = "estimator_name"
MODEL_DIR = "model"


pytestmark = pytest.mark.large


def get_iris():
    iris = sklearn.datasets.load_iris()
    return iris.data[:, :2], iris.target


def fit_model(model, X, y, fit_func_name):
    if fit_func_name == "fit":
        model.fit(X, y)

    if fit_func_name == "fit_transform":
        model.fit_transform(X, y)

    if fit_func_name == "fit_predict":
        model.fit_predict(X, y)

    return model


def get_run(run_id):
    return mlflow.tracking.MlflowClient().get_run(run_id)


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    return data.params, data.metrics, tags, artifacts


def load_model_by_run_id(run_id):
    return mlflow.sklearn.load_model("runs:/{}/{}".format(run_id, MODEL_DIR))


def get_model_conf(artifact_uri, model_subpath=MODEL_DIR):
    model_conf_path = os.path.join(artifact_uri, model_subpath, "MLmodel")
    return Model.load(model_conf_path)


def stringify_dict_values(d):
    return {k: str(v) for k, v in d.items()}


def truncate_dict(d):
    return _truncate_dict(d, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)


def get_expected_class_tags(model):
    return {
        ESTIMATOR_NAME: model.__class__.__name__,
        ESTIMATOR_CLASS: model.__class__.__module__ + "." + model.__class__.__name__,
    }


def assert_predict_equal(left, right, X):
    np.testing.assert_array_equal(left.predict(X), right.predict(X))


@pytest.fixture(params=FIT_FUNC_NAMES)
def fit_func_name(request):
    return request.param


@pytest.fixture(autouse=True, scope="function")
def force_try_mlflow_log_to_fail(request):
    # autolog contains multiple `try_mlflow_log`. They unexpectedly allow tests that
    # should fail to pass (without us noticing). To prevent that, temporarily turns
    # warnings emitted by `try_mlflow_log` into errors.
    if "disable_force_try_mlflow_log_to_fail" in request.keywords:
        yield
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=r"^Logging to MLflow failed", category=UserWarning,
            )
            yield


@pytest.mark.xfail(strict=True, raises=UserWarning)
def test_force_try_mlflow_log_to_fail():
    with mlflow.start_run():
        try_mlflow_log(lambda: 1 / 0)


@pytest.mark.disable_force_try_mlflow_log_to_fail
def test_no_force_try_mlflow_log_to_fail():
    with mlflow.start_run():
        try_mlflow_log(lambda: 1 / 0)


def test_autolog_preserves_original_function_attributes():
    def get_func_attrs(f):
        attrs = {}
        for attr_name in ["__doc__", "__name__"]:
            if hasattr(f, attr_name):
                attrs[attr_name] = getattr(f, attr_name)

        attrs["__signature__"] = inspect.signature(f)
        return attrs

    def get_cls_attrs(cls):
        attrs = {}
        for method_name in FIT_FUNC_NAMES:
            if hasattr(cls, method_name):
                attr = getattr(cls, method_name)
                if isinstance(attr, property):
                    continue

                attrs[method_name] = get_func_attrs(attr)
        return attrs

    before = [get_cls_attrs(cls) for _, cls in mlflow.sklearn.utils._all_estimators()]
    mlflow.sklearn.autolog()
    after = [get_cls_attrs(cls) for _, cls in mlflow.sklearn.utils._all_estimators()]

    for b, a in zip(before, after):
        assert b == a


@pytest.mark.skipif(
    _is_supported_version(), reason="This test fails on supported versions of sklearn"
)
def test_autolog_emits_warning_on_unsupported_versions_of_sklearn():
    with pytest.warns(
        UserWarning, match="Autologging utilities may not work properly on scikit-learn"
    ):
        mlflow.sklearn.autolog()


def test_autolog_does_not_terminate_active_run():
    mlflow.sklearn.autolog()
    mlflow.start_run()
    sklearn.cluster.KMeans().fit(*get_iris())
    assert mlflow.active_run() is not None
    mlflow.end_run()


def test_estimator(fit_func_name):
    mlflow.sklearn.autolog()

    # use `KMeans` because it implements `fit`, `fit_transform`, and `fit_predict`.
    model = sklearn.cluster.KMeans()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = fit_model(model, X, y, fit_func_name)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts

    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)


def test_meta_estimator():
    mlflow.sklearn.autolog()

    estimators = [
        ("std_scaler", sklearn.preprocessing.StandardScaler()),
        ("svc", sklearn.svm.SVC()),
    ]
    model = sklearn.pipeline.Pipeline(estimators)
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_get_params_returns_dict_that_has_more_keys_than_max_params_tags_per_batch():
    mlflow.sklearn.autolog()

    large_params = {str(i): str(i) for i in range(MAX_PARAMS_TAGS_PER_BATCH + 1)}
    X, y = get_iris()

    with mock.patch("sklearn.cluster.KMeans.get_params", return_value=large_params):
        with mlflow.start_run() as run:
            model = sklearn.cluster.KMeans()
            model.fit(X, y)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run.info.run_id)
    assert params == large_params
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)


@pytest.mark.parametrize(
    "long_params, messages",
    [
        # key exceeds the limit
        ({("a" * (MAX_ENTITY_KEY_LENGTH + 1)): "b"}, ["Truncated the key"]),
        # value exceeds the limit
        ({"a": "b" * (MAX_PARAM_VAL_LENGTH + 1)}, ["Truncated the value"]),
        # both key and value exceed the limit
        (
            {("a" * (MAX_ENTITY_KEY_LENGTH + 1)): "b" * (MAX_PARAM_VAL_LENGTH + 1)},
            ["Truncated the key", "Truncated the value"],
        ),
    ],
)
def test_get_params_returns_dict_whose_key_or_value_exceeds_length_limit(long_params, messages):
    mlflow.sklearn.autolog()

    X, y = get_iris()

    with mock.patch("sklearn.cluster.KMeans.get_params", return_value=long_params), mock.patch(
        "mlflow.sklearn.utils._logger.warning"
    ) as mock_warning, mlflow.start_run() as run:
        model = sklearn.cluster.KMeans()
        model.fit(X, y)

    for idx, msg in enumerate(messages):
        assert mock_warning.call_args_list[idx].startswith(msg)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run.info.run_id)
    assert params == truncate_dict(long_params)
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)


@pytest.mark.parametrize("Xy_passed_as", ["only_y_kwarg", "both_kwarg", "both_kwargs_swapped"])
def test_fit_takes_Xy_as_keyword_arguments(Xy_passed_as):
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()
    X, y = get_iris()

    with mlflow.start_run() as run:
        if Xy_passed_as == "only_y_kwarg":
            model.fit(X, y=y)
        elif Xy_passed_as == "both_kwarg":
            model.fit(X=X, y=y)
        elif Xy_passed_as == "both_kwargs_swapped":
            model.fit(y=y, X=X)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_call_fit_with_arguments_score_does_not_accept():
    mlflow.sklearn.autolog()

    from sklearn.linear_model import SGDRegressor

    assert "intercept_init" in _get_arg_names(SGDRegressor.fit)
    assert "intercept_init" not in _get_arg_names(SGDRegressor.score)

    mock_obj = mock.Mock()

    def mock_score(self, X, y, sample_weight=None):  # pylint: disable=unused-argument
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(SGDRegressor.score) == inspect.signature(mock_score)

    SGDRegressor.score = mock_score
    model = SGDRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y, intercept_init=0)
        mock_obj.assert_called_once_with(X, y, None)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


@pytest.mark.parametrize("sample_weight_passed_as", ["positional", "keyword"])
def test_both_fit_and_score_contain_sample_weight(sample_weight_passed_as):
    mlflow.sklearn.autolog()

    from sklearn.linear_model import SGDRegressor

    # ensure that we use an appropriate model for this test
    assert "sample_weight" in _get_arg_names(SGDRegressor.fit)
    assert "sample_weight" in _get_arg_names(SGDRegressor.score)

    mock_obj = mock.Mock()

    def mock_score(self, X, y, sample_weight=None):  # pylint: disable=unused-argument
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(SGDRegressor.score) == inspect.signature(mock_score)

    SGDRegressor.score = mock_score
    model = SGDRegressor()
    X, y = get_iris()
    sample_weight = abs(np.random.randn(len(X)))

    with mlflow.start_run() as run:
        if sample_weight_passed_as == "positional":
            model.fit(X, y, None, None, sample_weight)
        elif sample_weight_passed_as == "keyword":
            model.fit(X, y, sample_weight=sample_weight)
        mock_obj.assert_called_once_with(X, y, sample_weight)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_only_fit_contains_sample_weight():
    mlflow.sklearn.autolog()

    from sklearn.linear_model import RANSACRegressor

    assert "sample_weight" in _get_arg_names(RANSACRegressor.fit)
    assert "sample_weight" not in _get_arg_names(RANSACRegressor.score)

    mock_obj = mock.Mock()

    def mock_score(self, X, y):  # pylint: disable=unused-argument
        mock_obj(X, y)
        return 0

    assert inspect.signature(RANSACRegressor.score) == inspect.signature(mock_score)

    RANSACRegressor.score = mock_score
    model = RANSACRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)
        mock_obj.assert_called_once_with(X, y)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_only_score_contains_sample_weight():
    mlflow.sklearn.autolog()

    from sklearn.gaussian_process import GaussianProcessRegressor

    assert "sample_weight" not in _get_arg_names(GaussianProcessRegressor.fit)
    assert "sample_weight" in _get_arg_names(GaussianProcessRegressor.score)

    mock_obj = mock.Mock()

    def mock_score(self, X, y, sample_weight=None):  # pylint: disable=unused-argument
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(GaussianProcessRegressor.score) == inspect.signature(mock_score)

    GaussianProcessRegressor.score = mock_score
    model = GaussianProcessRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)
        mock_obj.assert_called_once_with(X, y, None)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert metrics == {TRAINING_SCORE: model.score(X, y)}
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_autolog_terminates_run_when_active_run_does_not_exist_and_fit_fails():
    mlflow.sklearn.autolog()

    with pytest.raises(ValueError, match="Penalty term must be positive"):
        sklearn.svm.LinearSVC(C=-1).fit(*get_iris())

    latest_run = mlflow.search_runs().iloc[0]
    assert mlflow.active_run() is None
    assert latest_run.status == "FAILED"


def test_autolog_does_not_terminate_run_when_active_run_exists_and_fit_fails():
    mlflow.sklearn.autolog()
    run = mlflow.start_run()

    with pytest.raises(ValueError, match="Penalty term must be positive"):
        sklearn.svm.LinearSVC(C=-1).fit(*get_iris())

    assert mlflow.active_run() is not None
    assert mlflow.active_run() is run
    mlflow.end_run()


def test_autolog_emits_warning_message_when_score_fails():
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()

    @functools.wraps(model.score)
    def throwing_score(X, y=None, sample_weight=None):  # pylint: disable=unused-argument
        raise Exception("EXCEPTION")

    model.score = throwing_score

    with mlflow.start_run() as run, mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        model.fit(*get_iris())
        mock_warning.assert_called_once()
        mock_warning.called_once_with(
            "KMeans.score failed. The 'training_score' metric will not be recorded. "
            "Scoring error: EXCEPTION"
        )

    metrics = get_run_data(run.info.run_id)[1]
    assert metrics == {}


def test_fit_xxx_performs_logging_only_once(fit_func_name):
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()
    X, y = get_iris()

    with mock.patch("mlflow.log_params") as mock_log_params, mock.patch(
        "mlflow.log_metric"
    ) as mock_log_metric, mock.patch("mlflow.set_tags") as mock_set_tags, mock.patch(
        "mlflow.sklearn.log_model"
    ) as mock_log_model:

        with mlflow.start_run() as run:
            model = fit_model(model, X, y, fit_func_name)
            mock_log_params.assert_called_once()
            mock_log_metric.assert_called_once()
            mock_set_tags.assert_called_once()
            mock_log_model.assert_called_once()

        query = "tags.{} = '{}'".format(MLFLOW_PARENT_RUN_ID, run.info.run_id)
        assert len(mlflow.search_runs([run.info.experiment_id])) == 1
        assert len(mlflow.search_runs([run.info.experiment_id], query)) == 0


def test_meta_estimator_fit_performs_logging_only_once():
    mlflow.sklearn.autolog()

    estimators = [
        ("std_scaler", sklearn.preprocessing.StandardScaler()),
        ("svc", sklearn.svm.SVC()),
    ]
    model = sklearn.pipeline.Pipeline(estimators)
    X, y = get_iris()

    with mock.patch("mlflow.log_params") as mock_log_params, mock.patch(
        "mlflow.log_metric"
    ) as mock_log_metric, mock.patch("mlflow.set_tags") as mock_set_tags, mock.patch(
        "mlflow.sklearn.log_model"
    ) as mock_log_model:

        with mlflow.start_run() as run:
            model.fit(X, y)
            mock_log_params.assert_called_once()
            mock_log_metric.assert_called_once()
            mock_set_tags.assert_called_once()
            mock_log_model.assert_called_once()

        query = "tags.{} = '{}'".format(MLFLOW_PARENT_RUN_ID, run.info.run_id)
        assert len(mlflow.search_runs([run.info.experiment_id])) == 1
        assert len(mlflow.search_runs([run.info.experiment_id], query)) == 0


@pytest.mark.parametrize(
    "cv_class, search_space",
    [
        (sklearn.model_selection.GridSearchCV, {"kernel": ("linear", "rbf"), "C": [1, 5, 10]}),
        (sklearn.model_selection.RandomizedSearchCV, {"C": uniform(loc=0, scale=4)}),
    ],
)
@pytest.mark.parametrize("backend", [None, "threading", "loky"])
def test_parameter_search_estimators_produce_expected_outputs(cv_class, search_space, backend):
    mlflow.sklearn.autolog()

    svc = sklearn.svm.SVC()
    cv_model = cv_class(svc, search_space, n_jobs=5, return_train_score=True)
    X, y = get_iris()

    def train_cv_model():
        if backend is None:
            cv_model.fit(X, y)
        else:
            with sklearn.utils.parallel_backend(backend=backend):
                cv_model.fit(X, y)

    with mlflow.start_run() as run:
        train_cv_model()
        run_id = run.info.run_id

    params, metrics, tags, artifacts = get_run_data(run_id)
    expected_cv_params = truncate_dict(stringify_dict_values(cv_model.get_params(deep=False)))
    expected_cv_params.update(
        {
            "best_{}".format(param_name): str(param_value)
            for param_name, param_value in cv_model.best_params_.items()
        }
    )
    assert params == expected_cv_params
    assert metrics == {TRAINING_SCORE: cv_model.score(X, y)}
    assert tags == get_expected_class_tags(cv_model)
    assert MODEL_DIR in artifacts
    assert "best_estimator" in artifacts
    assert "cv_results.csv" in artifacts

    best_estimator = mlflow.sklearn.load_model("runs:/{}/best_estimator".format(run_id))
    assert isinstance(best_estimator, sklearn.svm.SVC)
    cv_model = mlflow.sklearn.load_model("runs:/{}/{}".format(run_id, MODEL_DIR))
    assert isinstance(cv_model, cv_class)

    # Ensure that a signature and input example are produced for the best estimator
    best_estimator_conf = get_model_conf(run.info.artifact_uri, "best_estimator")
    assert best_estimator_conf.signature == infer_signature(X, best_estimator.predict(X[:5]))

    best_estimator_path = os.path.join(run.info.artifact_uri, "best_estimator")
    input_example = _read_example(best_estimator_conf, best_estimator_path)
    best_estimator.predict(input_example)  # Ensure that input example evaluation succeeds

    client = mlflow.tracking.MlflowClient()
    child_runs = client.search_runs(
        run.info.experiment_id, "tags.`mlflow.parentRunId` = '{}'".format(run_id)
    )
    cv_results = pd.DataFrame.from_dict(cv_model.cv_results_)
    # We expect to have created a child run for each point in the parameter search space
    assert len(child_runs) == len(cv_results)

    # Verify that each set of parameter search results has a corresponding MLflow run
    # with the expected data
    for _, result in cv_results.iterrows():
        result_params = result.get("params", {})
        params_search_clause = " and ".join(
            ["params.`{}` = '{}'".format(key, value) for key, value in result_params.items()]
        )
        search_filter = "tags.`mlflow.parentRunId` = '{}' and {}".format(
            run_id, params_search_clause
        )
        child_runs = client.search_runs(run.info.experiment_id, search_filter)
        assert len(child_runs) == 1
        child_run = child_runs[0]
        assert child_run.info.status == RunStatus.to_string(RunStatus.FINISHED)
        _, child_metrics, child_tags, _ = get_run_data(child_run.info.run_id)
        assert child_tags == get_expected_class_tags(svc)
        assert "mean_test_score" in child_metrics.keys()
        assert "std_test_score" in child_metrics.keys()
        # Ensure that we do not capture separate metrics for each cross validation split, which
        # would produce very noisy metrics results
        assert len([metric for metric in child_metrics.keys() if metric.startswith("split")]) == 0


def test_parameter_search_handles_large_volume_of_metric_outputs():
    mlflow.sklearn.autolog()

    metrics_size = MAX_METRICS_PER_BATCH + 10
    metrics_to_log = {
        "score_{}".format(i): sklearn.metrics.make_scorer(lambda y, y_pred, **kwargs: 10)
        for i in range(metrics_size)
    }

    with mlflow.start_run() as run:
        svc = sklearn.svm.SVC()
        cv_model = sklearn.model_selection.GridSearchCV(
            svc, {"C": [1]}, n_jobs=1, scoring=metrics_to_log, refit=False
        )
        cv_model.fit(*get_iris())
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    child_runs = client.search_runs(
        run.info.experiment_id, "tags.`mlflow.parentRunId` = '{}'".format(run_id)
    )
    assert len(child_runs) == 1
    child_run = child_runs[0]

    assert len(child_run.data.metrics) >= metrics_size


@pytest.mark.disable_force_try_mlflow_log_to_fail
@pytest.mark.parametrize(
    "failing_specialization",
    [
        "mlflow.sklearn.utils._log_parameter_search_results_as_artifact",
        "mlflow.sklearn.utils._create_child_runs_for_parameter_search",
    ],
)
def test_autolog_does_not_throw_when_parameter_search_logging_fails(failing_specialization):
    with mock.patch(failing_specialization, side_effect=Exception("Failed")) as mock_func:
        # Enable autologging after mocking the parameter search specialization function
        # to ensure that the mock is applied before the function is imported
        mlflow.sklearn.autolog()
        svc = sklearn.svm.SVC()
        cv_model = sklearn.model_selection.GridSearchCV(svc, {"C": [1]}, n_jobs=1)
        cv_model.fit(*get_iris())
        mock_func.assert_called_once()


@pytest.mark.disable_force_try_mlflow_log_to_fail
@pytest.mark.parametrize(
    "func_to_fail",
    ["mlflow.log_params", "mlflow.log_metric", "mlflow.set_tags", "mlflow.sklearn.log_model"],
)
def test_autolog_does_not_throw_when_mlflow_logging_fails(func_to_fail):
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()
    X, y = get_iris()

    with mlflow.start_run(), mock.patch(
        func_to_fail, side_effect=Exception(func_to_fail)
    ) as mock_func:

        model.fit(X, y)
        mock_func.assert_called_once()


@pytest.mark.parametrize("data_type", [pd.DataFrame, np.array])
def test_autolog_logs_signature_and_input_example(data_type):
    mlflow.sklearn.autolog()

    X, y = get_iris()
    X = data_type(X)
    y = data_type(y)
    model = sklearn.linear_model.LinearRegression()

    with mlflow.start_run() as run:
        model.fit(X, y)
        model_path = os.path.join(run.info.artifact_uri, MODEL_DIR)

    model_conf = get_model_conf(run.info.artifact_uri)
    input_example = _read_example(model_conf, model_path)
    pyfunc_model = mlflow.pyfunc.load_model(model_path)

    assert model_conf.signature == infer_signature(X, model.predict(X[:5]))
    np.testing.assert_array_equal(pyfunc_model.predict(input_example), model.predict(X[:5]))


def test_autolog_does_not_throw_when_failing_to_sample_X():
    class ArrayThatThrowsWhenSliced(np.ndarray):
        def __new__(cls, input_array):
            return np.asarray(input_array).view(cls)

        def __getitem__(self, key):
            if isinstance(key, slice):
                raise IndexError("DO NOT SLICE ME")
            return super(ArrayThatThrowsWhenSliced, self).__getitem__(key)

    X, y = get_iris()
    throwing_X = ArrayThatThrowsWhenSliced(X)

    # ensure throwing_X throws when sliced
    with pytest.raises(IndexError, match="DO NOT SLICE ME"):
        _ = throwing_X[:5]

    mlflow.sklearn.autolog()
    model = sklearn.linear_model.LinearRegression()

    with mlflow.start_run() as run, mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        model.fit(throwing_X, y)

    model_conf = get_model_conf(run.info.artifact_uri)

    mock_warning.assert_called_once()
    mock_warning.call_args[0][0].endswith("DO NOT SLICE ME")
    assert "signature" not in model_conf.to_dict()
    assert "saved_input_example_info" not in model_conf.to_dict()


def test_autolog_logs_signature_and_input_example_only_when_estimator_defines_predict():
    from sklearn.cluster import AgglomerativeClustering

    mlflow.sklearn.autolog()

    X, y = get_iris()
    model = AgglomerativeClustering()
    assert not hasattr(model, "predict")

    with mlflow.start_run() as run:
        model.fit(X, y)

    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" not in model_conf.to_dict()
    assert "saved_input_example_info" not in model_conf.to_dict()


def test_autolog_does_not_throw_when_predict_fails():
    X, y = get_iris()

    # Note that `mock_warning` will be called twice because if `predict` throws, `score` also throws
    with mlflow.start_run() as run, mock.patch(
        "sklearn.linear_model.LinearRegression.predict", side_effect=Exception("Failed")
    ), mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        mlflow.sklearn.autolog()
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

    mock_warning.assert_called_with("Failed to infer an input example and model signature: Failed")
    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" not in model_conf.to_dict()
    assert "saved_input_example_info" not in model_conf.to_dict()


def test_autolog_does_not_throw_when_infer_signature_fails():
    X, y = get_iris()

    with mlflow.start_run() as run, mock.patch(
        "mlflow.models.infer_signature", side_effect=Exception("Failed")
    ), mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        mlflow.sklearn.autolog()
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

    mock_warning.assert_called_once_with(
        "Failed to infer an input example and model signature: Failed"
    )
    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" not in model_conf.to_dict()
    assert "saved_input_example_info" not in model_conf.to_dict()
