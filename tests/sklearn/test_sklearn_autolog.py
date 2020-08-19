import functools
import inspect
from mock import mock

import numpy as np
import pytest
import sklearn
import sklearn.datasets as datasets

import mlflow.sklearn
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.sklearn.utils import _get_arg_names

FIT_FUNC_NAMES = ["fit", "fit_transform", "fit_predict"]
TRAINING_SCORE = "training_score"
ESTIMATOR_CLASS = "estimator_class"
ESTIMATOR_NAME = "estimator_name"


def get_iris():
    iris = datasets.load_iris()
    return iris.data[:, :2], iris.target


def fit_model(model, Xy, fit_func_name):
    if fit_func_name == "fit":
        model.fit(*Xy)

    if fit_func_name == "fit_transform":
        model.fit_transform(*Xy)

    if fit_func_name == "fit_predict":
        model.fit_predict(*Xy)

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
    return mlflow.sklearn.load_model("runs:/{}/model".format(run_id))


def stringify_dict_values(d):
    return {k: str(v) for k, v in d.items()}


@pytest.fixture(params=FIT_FUNC_NAMES)
def fit_func_name(request):
    return request.param


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
                attrs[method_name] = get_func_attrs(getattr(cls, method_name))
        return attrs

    before = [get_cls_attrs(cls) for _, cls in mlflow.sklearn._get_all_estimators()]
    mlflow.sklearn.autolog()
    after = [get_cls_attrs(cls) for _, cls in mlflow.sklearn._get_all_estimators()]

    for b, a in zip(before, after):
        assert b == a


@pytest.mark.skipif(
    not mlflow.sklearn._is_old_version(), reason="This test fails on sklearn>=0.20.3"
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
    Xy = get_iris()

    with mlflow.start_run() as run:
        run_id = run._info.run_id
        model = fit_model(model, Xy, fit_func_name)

    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == stringify_dict_values(model.get_params(deep=True))
    assert metrics == {TRAINING_SCORE: model.score(*Xy)}
    assert tags == {
        ESTIMATOR_NAME: model.__class__.__name__,
        ESTIMATOR_CLASS: model.__class__.__module__ + "." + model.__class__.__name__,
    }
    assert "model" in artifacts

    loaded_model = load_model_by_run_id(run_id)
    np.testing.assert_array_equal(loaded_model.predict(Xy[0]), model.predict(Xy[0]))


def test_meta_estimator():
    mlflow.sklearn.autolog()

    estimators = [
        ("std_scaler", sklearn.preprocessing.StandardScaler()),
        ("svc", sklearn.svm.SVC()),
    ]
    model = sklearn.pipeline.Pipeline(estimators)
    Xy = get_iris()

    with mlflow.start_run() as run:
        run_id = run._info.run_id
        model = fit_model(model, Xy, "fit")

    params, metrics, tags, artifacts = get_run_data(run._info.run_id)
    assert params == stringify_dict_values(model.get_params(deep=True))
    assert metrics == {TRAINING_SCORE: model.score(*Xy)}
    assert tags == {
        ESTIMATOR_NAME: model.__class__.__name__,
        ESTIMATOR_CLASS: model.__class__.__module__ + "." + model.__class__.__name__,
    }
    assert "model" in artifacts

    loaded_model = load_model_by_run_id(run_id)
    np.testing.assert_array_equal(loaded_model.predict(Xy[0]), model.predict(Xy[0]))


def test_get_params_returns_dict_with_more_than_100_top_level_keys():
    mlflow.sklearn.autolog()

    large_params = {str(i): str(i) for i in range(101)}

    with mock.patch("sklearn.cluster.KMeans.get_params", return_value=large_params):
        with mlflow.start_run() as run:
            Xy = get_iris()
            model = sklearn.cluster.KMeans()
            run_id = run._info.run_id
            model = fit_model(model, Xy, "fit")

    params, metrics, tags, artifacts = get_run_data(run._info.run_id)
    assert params == large_params
    assert metrics == {TRAINING_SCORE: model.score(*Xy)}
    assert tags == {
        ESTIMATOR_NAME: model.__class__.__name__,
        ESTIMATOR_CLASS: model.__class__.__module__ + "." + model.__class__.__name__,
    }
    assert "model" in artifacts

    loaded_model = load_model_by_run_id(run_id)
    np.testing.assert_array_equal(loaded_model.predict(Xy[0]), model.predict(Xy[0]))


def test_call_fit_with_arguments_score_does_not_accept():
    mlflow.sklearn.autolog()

    model = sklearn.linear_model.SGDRegressor()

    assert "intercept_init" in _get_arg_names(model.fit)
    assert "intercept_init" not in _get_arg_names(model.score)

    with mlflow.start_run():
        Xy = get_iris()
        model.fit(*Xy, intercept_init=0)


def test_both_fit_and_score_contain_sample_weight():
    mlflow.sklearn.autolog()

    model = sklearn.linear_model.SGDRegressor()

    # ensure that we use an appropriate model for this test
    assert "sample_weight" in _get_arg_names(model.fit)
    assert "sample_weight" in _get_arg_names(model.score)

    mock_obj = mock.Mock()

    def mock_score(X, y, sample_weight=None):
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(model.score) == inspect.signature(mock_score)

    model.score = mock_score

    with mlflow.start_run():
        Xy = get_iris()
        sample_weight = abs(np.random.randn(len(Xy[0])))
        model.fit(*Xy, sample_weight=sample_weight)
        mock_obj.assert_called_once_with(*Xy, sample_weight)


def test_only_fit_contains_sample_weight():
    mlflow.sklearn.autolog()

    model = sklearn.linear_model.RANSACRegressor()

    assert "sample_weight" in _get_arg_names(model.fit)
    assert "sample_weight" not in _get_arg_names(model.score)

    mock_obj = mock.Mock()

    def mock_score(X, y):
        mock_obj(X, y)
        return 0

    assert inspect.signature(model.score) == inspect.signature(mock_score)

    model.score = mock_score

    with mlflow.start_run():
        Xy = get_iris()
        model.fit(*Xy)
        mock_obj.assert_called_once_with(*Xy)


def test_only_score_contains_sample_weight():
    mlflow.sklearn.autolog()

    model = sklearn.gaussian_process.GaussianProcessRegressor()

    assert "sample_weight" not in _get_arg_names(model.fit)
    assert "sample_weight" in _get_arg_names(model.score)

    mock_obj = mock.Mock()

    def mock_score(X, y, sample_weight=None):
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(model.score) == inspect.signature(mock_score)

    model.score = mock_score

    with mlflow.start_run():
        Xy = get_iris()
        model.fit(*Xy)
        mock_obj.assert_called_once_with(*Xy, None)


def test_autolog_terminates_run_when_active_run_does_not_exist_and_fit_fails():
    mlflow.sklearn.autolog()

    with pytest.raises(ValueError, match="Penalty term must be positive"):
        sklearn.svm.LinearSVC(C=-1).fit(*get_iris())

    latest_run = mlflow.search_runs().iloc[0]
    assert mlflow.active_run() is None
    assert latest_run.status == "FAILED"


def test_autolog_does_not_terminate_run_but_throws_instead_when_active_run_exists_and_fit_fails():
    mlflow.sklearn.autolog()
    run = mlflow.start_run()

    with pytest.raises(ValueError, match="Penalty term must be positive"):
        sklearn.svm.LinearSVC(C=-1).fit(*get_iris())

    assert mlflow.active_run() is not None
    assert mlflow.active_run() is run
    mlflow.end_run()


def test_autolog_emits_warning_message_when_score_fails():
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run, mock.patch("logging.Logger.warning") as mock_warning:
        model = sklearn.cluster.KMeans()

        @functools.wraps(model.score)
        def throwing_score(X, y=None, sample_weight=None):
            raise Exception("EXCEPTION")

        model.score = throwing_score
        model.fit(*get_iris())

    metrics = get_run_data(run._info.run_id)[1]
    assert metrics == {}
    mock_warning.assert_called_once()
    assert mock_warning.call_args[0][0] == ("KMeans.score failed: EXCEPTION")


def test_fit_xxx_performs_logging_only_once(fit_func_name):
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()
    Xy = get_iris()

    with mock.patch("mlflow.log_params") as mock_log_params, mock.patch(
        "mlflow.log_metric"
    ) as mock_log_metric, mock.patch("mlflow.set_tags") as mock_set_tags, mock.patch(
        "mlflow.sklearn.log_model"
    ) as mock_log_model:

        with mlflow.start_run() as run:
            model = fit_model(model, Xy, fit_func_name)
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
    Xy = get_iris()

    with mock.patch("mlflow.log_params") as mock_log_params, mock.patch(
        "mlflow.log_metric"
    ) as mock_log_metric, mock.patch("mlflow.set_tags") as mock_set_tags, mock.patch(
        "mlflow.sklearn.log_model"
    ) as mock_log_model:

        with mlflow.start_run() as run:
            model = fit_model(model, Xy, "fit")
            mock_log_params.assert_called_once()
            mock_log_metric.assert_called_once()
            mock_set_tags.assert_called_once()
            mock_log_model.assert_called_once()

        query = "tags.{} = '{}'".format(MLFLOW_PARENT_RUN_ID, run._info.run_id)
        assert len(mlflow.search_runs([run._info.experiment_id])) == 1
        assert len(mlflow.search_runs([run._info.experiment_id], query)) == 0
