import functools
import inspect
from mock import mock
import warnings

import numpy as np
import pytest
import sklearn
import sklearn.datasets

import mlflow.sklearn
from mlflow.sklearn.utils import _MIN_SKLEARN_VERSION, _get_arg_names, _truncate_dict
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.autologging_utils import try_mlflow_log
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_PARAM_KEY_LENGTH,
    MAX_PARAM_VAL_LENGTH,
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


def stringify_dict_values(d):
    return {k: str(v) for k, v in d.items()}


def truncate_dict(d):
    return _truncate_dict(d, MAX_PARAM_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)


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
def throw_if_try_mlflow_log_has_emitted_warnings():
    # autolog contains multiple try_mlflow_log. They may hide unintentional errors (often caused by
    # incorrect test settings) and allow tests that should not pass to pass (without us noticing).
    # To prevent that, throw if try_mlflow_log has emitted warnings during test execution.
    with warnings.catch_warnings(record=True) as ws:
        yield
        mlflow_warnings = [w for w in ws if str(w.message).startswith("Logging to MLflow failed")]
        assert len(mlflow_warnings) == 0


@pytest.mark.xfail
def test_throw_if_try_mlflow_log_has_emitted_warnings():
    with mlflow.start_run():
        # This line should emit a warning because a metric value must be a float value.
        try_mlflow_log(mlflow.log_metric, "m", "m")


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
    not mlflow.sklearn.utils._is_old_version(),
    reason="This test fails on sklearn>={}".format(_MIN_SKLEARN_VERSION),
)
def test_autolog_emits_warning_on_older_versions_of_sklearn():
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

    run_id = run._info.run_id
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

    run_id = run._info.run_id
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

    run_id = run._info.run_id
    params, metrics, tags, artifacts = get_run_data(run._info.run_id)
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
        ({("a" * (MAX_PARAM_KEY_LENGTH + 1)): "b"}, ["Truncated the key"]),
        # value exceeds the limit
        ({"a": "b" * (MAX_PARAM_VAL_LENGTH + 1)}, ["Truncated the value"]),
        # both key and value exceed the limit
        (
            {("a" * (MAX_PARAM_KEY_LENGTH + 1)): "b" * (MAX_PARAM_VAL_LENGTH + 1)},
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

    run_id = run._info.run_id
    params, metrics, tags, artifacts = get_run_data(run._info.run_id)
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

    run_id = run._info.run_id
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

    def mock_score(self, X, y, sample_weight=None):
        # pylint: disable=unused-argument
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(SGDRegressor.score) == inspect.signature(mock_score)

    SGDRegressor.score = mock_score
    model = SGDRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y, intercept_init=0)
        mock_obj.assert_called_once_with(X, y, None)

    run_id = run._info.run_id
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

    def mock_score(self, X, y, sample_weight=None):
        # pylint: disable=unused-argument
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

    run_id = run._info.run_id
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

    def mock_score(self, X, y):
        # pylint: disable=unused-argument
        mock_obj(X, y)
        return 0

    assert inspect.signature(RANSACRegressor.score) == inspect.signature(mock_score)

    RANSACRegressor.score = mock_score
    model = RANSACRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)
        mock_obj.assert_called_once_with(X, y)

    run_id = run._info.run_id
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

    def mock_score(self, X, y, sample_weight=None):
        # pylint: disable=unused-argument
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(GaussianProcessRegressor.score) == inspect.signature(mock_score)

    GaussianProcessRegressor.score = mock_score
    model = GaussianProcessRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)
        mock_obj.assert_called_once_with(X, y, None)

    run_id = run._info.run_id
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
    def throwing_score(X, y=None, sample_weight=None):
        # pylint: disable=unused-argument
        raise Exception("EXCEPTION")

    model.score = throwing_score

    with mlflow.start_run() as run, mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        model.fit(*get_iris())

    metrics = get_run_data(run._info.run_id)[1]
    assert metrics == {}
    mock_warning.assert_called_once()
    mock_warning.called_once_with(
        "KMeans.score failed. The 'training_score' metric will not be recorded. "
        "Scoring error: EXCEPTION"
    )


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

        query = "tags.{} = '{}'".format(MLFLOW_PARENT_RUN_ID, run._info.run_id)
        assert len(mlflow.search_runs([run._info.experiment_id])) == 1
        assert len(mlflow.search_runs([run._info.experiment_id], query)) == 0


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

        query = "tags.{} = '{}'".format(MLFLOW_PARENT_RUN_ID, run._info.run_id)
        assert len(mlflow.search_runs([run._info.experiment_id])) == 1
        assert len(mlflow.search_runs([run._info.experiment_id], query)) == 0
