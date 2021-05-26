import functools
import inspect
from unittest import mock
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import sklearn
import sklearn.base
import sklearn.datasets
import sklearn.model_selection
from scipy.stats import uniform

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.signature import infer_signature
from mlflow.models.utils import _read_example
import mlflow.sklearn
from mlflow.entities import RunStatus
from mlflow.sklearn.utils import (
    _is_supported_version,
    _is_metric_supported,
    _is_plotting_supported,
    _get_arg_names,
    _truncate_dict,
    _log_child_runs_info,
)
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_AUTOLOGGING
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)

from tests.autologging.fixtures import test_mode_off

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

    if fit_func_name == "fake":
        if isinstance(model, sklearn.linear_model.LinearRegression):
            model.coef_ = np.random.random(size=np.shape(X)[-1])
            model.intercept_ = 0

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


def test_autolog_throws_error_with_negative_max_tuning_runs():
    with pytest.raises(
        MlflowException, match="`max_tuning_runs` must be non-negative, instead got -1."
    ):
        mlflow.sklearn.autolog(max_tuning_runs=-1)


@pytest.mark.parametrize(
    "max_tuning_runs, total_runs, output_statment",
    [
        (0, 4, "Logging no runs, all will be omitted"),
        (0, 1, "Logging no runs, one run will be omitted"),
        (1, 1, "Logging the best run, no runs will be omitted"),
        (5, 4, "Logging all runs, no runs will be omitted"),
        (4, 4, "Logging all runs, no runs will be omitted"),
        (2, 5, "Logging the 2 best runs, 3 runs will be omitted"),
    ],
)
def test_autolog_max_tuning_runs_logs_info_correctly(max_tuning_runs, total_runs, output_statment):
    with mock.patch("mlflow.sklearn.utils._logger.info") as mock_info:
        _log_child_runs_info(max_tuning_runs, total_runs)
        mock_info.assert_called_once()
        mock_info.called_once_with(output_statment)


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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts

    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)


def test_classifier_binary():
    mlflow.sklearn.autolog()
    # use RandomForestClassifier that has method [predict_proba], so that we can test
    # logging of (1) log_loss and (2) roc_auc_score.
    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)

    # use binary datasets to cover the test for roc curve & precision recall curve
    X, y_true = sklearn.datasets.load_breast_cancer(return_X_y=True)

    with mlflow.start_run() as run:
        model = fit_model(model, X, y_true, "fit")

    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)
    # For binary classification, y_score only accepts the probability of greater label
    y_pred_prob_roc = y_pred_prob[:, 1]

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))

    expected_metrics = {
        TRAINING_SCORE: model.score(X, y_true),
        "training_accuracy_score": sklearn.metrics.accuracy_score(y_true, y_pred),
        "training_precision_score": sklearn.metrics.precision_score(
            y_true, y_pred, average="weighted"
        ),
        "training_recall_score": sklearn.metrics.recall_score(y_true, y_pred, average="weighted"),
        "training_f1_score": sklearn.metrics.f1_score(y_true, y_pred, average="weighted"),
        "training_log_loss": sklearn.metrics.log_loss(y_true, y_pred_prob),
    }
    if _is_metric_supported("roc_auc_score"):
        expected_metrics["training_roc_auc_score"] = sklearn.metrics.roc_auc_score(
            y_true, y_score=y_pred_prob_roc, average="weighted", multi_class="ovo",
        )

    assert metrics == expected_metrics

    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts

    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    plot_names = []
    if _is_plotting_supported():
        plot_names.extend(
            [
                "{}.png".format("training_confusion_matrix"),
                "{}.png".format("training_roc_curve"),
                "{}.png".format("training_precision_recall_curve"),
            ]
        )

    assert all(x in artifacts for x in plot_names)

    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)
    # verify no figure is open
    assert len(plt.get_fignums()) == 0


def test_classifier_multi_class():
    mlflow.sklearn.autolog()
    # use RandomForestClassifier that has method [predict_proba], so that we can test
    # logging of (1) log_loss and (2) roc_auc_score.
    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)

    # use multi-class datasets to verify that roc curve & precision recall curve care not recorded
    X, y_true = get_iris()

    with mlflow.start_run() as run:
        model = fit_model(model, X, y_true, "fit")

    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))

    expected_metrics = {
        TRAINING_SCORE: model.score(X, y_true),
        "training_accuracy_score": sklearn.metrics.accuracy_score(y_true, y_pred),
        "training_precision_score": sklearn.metrics.precision_score(
            y_true, y_pred, average="weighted"
        ),
        "training_recall_score": sklearn.metrics.recall_score(y_true, y_pred, average="weighted"),
        "training_f1_score": sklearn.metrics.f1_score(y_true, y_pred, average="weighted"),
        "training_log_loss": sklearn.metrics.log_loss(y_true, y_pred_prob),
    }
    if _is_metric_supported("roc_auc_score"):
        expected_metrics["training_roc_auc_score"] = sklearn.metrics.roc_auc_score(
            y_true, y_score=y_pred_prob, average="weighted", multi_class="ovo",
        )

    assert metrics == expected_metrics

    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts

    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    plot_names = []
    if _is_plotting_supported():
        plot_names = ["{}.png".format("training_confusion_matrix")]

    assert all(x in artifacts for x in plot_names)

    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)


def test_regressor():
    mlflow.sklearn.autolog()
    # use simple `LinearRegression`, which only implements `fit`.
    model = sklearn.linear_model.LinearRegression()
    X, y_true = get_iris()

    with mlflow.start_run() as run:
        model = fit_model(model, X, y_true, "fit")

    y_pred = model.predict(X)
    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))

    assert metrics == {
        TRAINING_SCORE: model.score(X, y_true),
        "training_mse": sklearn.metrics.mean_squared_error(y_true, y_pred),
        "training_rmse": np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)),
        "training_mae": sklearn.metrics.mean_absolute_error(y_true, y_pred),
        "training_r2_score": sklearn.metrics.r2_score(y_true, y_pred),
    }
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
        "mlflow.utils._logger.warning"
    ) as mock_warning, mlflow.start_run() as run:
        model = sklearn.cluster.KMeans()
        model.fit(X, y)

    for idx, msg in enumerate(messages):
        assert mock_warning.call_args_list[idx].startswith(msg)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run.info.run_id)
    assert params == truncate_dict(long_params)
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: model.score(X, y)}.items() <= metrics.items()
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

    with mlflow.start_run(), mock.patch("mlflow.sklearn.utils._logger.warning") as mock_warning:
        model.fit(*get_iris())
        mock_warning.assert_called_once()
        mock_warning.called_once_with(
            "KMeans.score failed. The 'training_score' metric will not be recorded. "
            "Scoring error: EXCEPTION"
        )


def test_autolog_emits_warning_message_when_metric_fails():
    """
    Take precision_score metric from SVC as an example to test metric logging failure
    """
    mlflow.sklearn.autolog()

    model = sklearn.svm.SVC()

    @functools.wraps(sklearn.metrics.precision_score)
    def throwing_metrics(y_true, y_pred):  # pylint: disable=unused-argument
        raise Exception("EXCEPTION")

    with mlflow.start_run(), mock.patch(
        "mlflow.sklearn.utils._logger.warning"
    ) as mock_warning, mock.patch("sklearn.metrics.precision_score", side_effect=throwing_metrics):
        model.fit(*get_iris())
        mock_warning.assert_called_once()
        mock_warning.called_once_with(
            "SVC.precision_score failed. "
            "The 'precision_score' metric will not be recorded. "
            "Metric error: EXCEPTION"
        )


def test_autolog_emits_warning_message_when_model_prediction_fails():
    """
    Take GridSearchCV as an example, whose base class is "classifier" and will go
    through classifier's metric logging. When refit=False, the model will never get
    refitted, while during the metric logging what ".predict()" expects is a fitted model.
    Thus, a warning will be logged.
    """
    from sklearn.exceptions import NotFittedError

    mlflow.sklearn.autolog()

    metrics_size = 2
    metrics_to_log = {
        "score_{}".format(i): sklearn.metrics.make_scorer(lambda y, y_pred, **kwargs: 10)
        for i in range(metrics_size)
    }

    with mlflow.start_run(), mock.patch("mlflow.sklearn.utils._logger.warning") as mock_warning:
        svc = sklearn.svm.SVC()
        cv_model = sklearn.model_selection.GridSearchCV(
            svc, {"C": [1]}, n_jobs=1, scoring=metrics_to_log, refit=False
        )
        cv_model.fit(*get_iris())

        # Ensure `cv_model.predict` fails with `NotFittedError`
        msg = (
            "This GridSearchCV instance was initialized with refit=False. "
            "predict is available only after refitting on the best parameters"
        )
        with pytest.raises(NotFittedError, match=msg):
            cv_model.predict([[0, 0, 0, 0]])

        # Count how many times `mock_warning` has been called on not-fitted `predict` failure
        call_count = len([args for args in mock_warning.call_args_list if msg in args[0][0]])
        # If `_is_plotting_supported` returns True (meaning sklearn version is >= 0.22.0),
        # `mock_warning` should have been called twice, once for metrics, once for artifacts.
        # Otherwise, only once for metrics.
        call_count_expected = 2 if mlflow.sklearn.utils._is_plotting_supported() else 1
        assert call_count == call_count_expected


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
@pytest.mark.parametrize("max_tuning_runs", [None, 3])
def test_parameter_search_estimators_produce_expected_outputs(
    cv_class, search_space, backend, max_tuning_runs
):
    mlflow.sklearn.autolog(
        log_input_examples=True, log_model_signatures=True, max_tuning_runs=max_tuning_runs,
    )

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
    assert {
        TRAINING_SCORE: cv_model.score(X, y),
        "best_cv_score": cv_model.best_score_,
    }.items() <= metrics.items()
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
    num_total_results = len(cv_results)
    if max_tuning_runs is None:
        cv_results_best_n_df = cv_results
        cv_results_rest_df = pd.DataFrame()
    else:
        num_rest = max(0, num_total_results - max_tuning_runs)
        cv_results_best_n_df = cv_results.nsmallest(max_tuning_runs, "rank_test_score")
        cv_results_rest_df = cv_results.nlargest(num_rest, "rank_test_score", keep="last")
        # We expect to have created a child run for each point in the parameter search space
        # up to max_tuning_runs.
        assert len(child_runs) == max_tuning_runs
        assert len(child_runs) + num_rest == num_total_results

    # Verify that the best max_tuning_runs of parameter search results
    # have a corresponding MLflow run with the expected data
    for _, result in cv_results_best_n_df.iterrows():
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
        assert child_run.data.tags.get(MLFLOW_AUTOLOGGING) == mlflow.sklearn.FLAVOR_NAME
        assert "mean_test_score" in child_metrics.keys()
        assert "std_test_score" in child_metrics.keys()
        # Ensure that we do not capture separate metrics for each cross validation split, which
        # would produce very noisy metrics results
        assert len([metric for metric in child_metrics.keys() if metric.startswith("split")]) == 0

    # Verify that the rest of the parameter search results do not have
    # a corresponding MLflow run.
    for _, result in cv_results_rest_df.iterrows():
        result_params = result.get("params", {})
        params_search_clause = " and ".join(
            ["params.`{}` = '{}'".format(key, value) for key, value in result_params.items()]
        )
        search_filter = "tags.`mlflow.parentRunId` = '{}' and {}".format(
            run_id, params_search_clause
        )
        child_runs = client.search_runs(run.info.experiment_id, search_filter)
        assert len(child_runs) == 0


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


@pytest.mark.usefixtures(test_mode_off.__name__)
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


@pytest.mark.usefixtures(test_mode_off.__name__)
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
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

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

    # On GitHub Actions, `pyfunc_model.predict` and `model.predict` sometimes return
    # slightly different results:
    #
    # >>> pyfunc_model.predict(input_example)
    # [[0.171504346208176  ]
    #  [0.34346150441640155]  <- diff
    #  [0.06895096846585114]  <- diff
    #  [0.05925789882165455]
    #  [0.03424907823290102]]
    #
    # >>> model.predict(X[:5])
    # [[0.171504346208176  ]
    #  [0.3434615044164018 ]  <- diff
    #  [0.06895096846585136]  <- diff
    #  [0.05925789882165455]
    #  [0.03424907823290102]]
    #
    # As a workaround, use `assert_array_almost_equal` instead of `assert_array_equal`
    np.testing.assert_array_almost_equal(pyfunc_model.predict(input_example), model.predict(X[:5]))


def test_autolog_does_not_throw_when_failing_to_sample_X():
    class ArrayThatThrowsWhenSliced(np.ndarray):
        def __new__(cls, input_array):
            return np.asarray(input_array).view(cls)

        def __getitem__(self, key):
            if isinstance(key, slice):
                raise IndexError("DO NOT SLICE ME")
            return super().__getitem__(key)

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


def test_autolog_logs_signature_only_when_estimator_defines_predict():
    from sklearn.cluster import AgglomerativeClustering

    mlflow.sklearn.autolog(log_model_signatures=True)

    X, y = get_iris()
    model = AgglomerativeClustering()
    assert not hasattr(model, "predict")

    with mlflow.start_run() as run:
        model.fit(X, y)

    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" not in model_conf.to_dict()


def test_autolog_does_not_throw_when_predict_fails():
    X, y = get_iris()

    # Note that `mock_warning` will be called twice because if `predict` throws, `score` also throws
    with mlflow.start_run() as run, mock.patch(
        "sklearn.linear_model.LinearRegression.predict", side_effect=Exception("Failed")
    ), mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

    mock_warning.assert_called_with("Failed to infer model signature: Failed")
    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" not in model_conf.to_dict()


def test_autolog_does_not_throw_when_infer_signature_fails():
    X, y = get_iris()

    with mlflow.start_run() as run, mock.patch(
        "mlflow.models.infer_signature", side_effect=Exception("Failed")
    ), mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

    mock_warning.assert_called_once_with("Failed to infer model signature: Failed")
    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" not in model_conf.to_dict()


@pytest.mark.large
@pytest.mark.parametrize("log_input_examples", [True, False])
@pytest.mark.parametrize("log_model_signatures", [True, False])
def test_autolog_configuration_options(log_input_examples, log_model_signatures):
    X, y = get_iris()

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog(
            log_input_examples=log_input_examples, log_model_signatures=log_model_signatures
        )
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)
    model_conf = get_model_conf(run.info.artifact_uri)
    assert ("saved_input_example_info" in model_conf.to_dict()) == log_input_examples
    assert ("signature" in model_conf.to_dict()) == log_model_signatures


@pytest.mark.large
@pytest.mark.parametrize("log_models", [True, False])
def test_sklearn_autolog_log_models_configuration(log_models):
    X, y = get_iris()

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog(log_models=log_models)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

    run_id = run.info.run_id
    _, _, _, artifacts = get_run_data(run_id)
    assert (MODEL_DIR in artifacts) == log_models


@pytest.mark.large
def test_autolog_does_not_capture_runs_for_preprocessing_or_feature_manipulation_estimators():
    """
    Verifies that preprocessing and feature manipulation estimators, which represent data
    manipulation steps (e.g., normalization, label encoding) rather than ML models, do not
    produce runs when their fit_* operations are invoked independently of an ML pipeline
    """
    mlflow.sklearn.autolog()

    # Create a run using the MLflow client, which will be resumed via the fluent API,
    # in order to avoid setting fluent-level tags (e.g., source and user). Suppressing these
    # tags simplifies test validation logic
    client = mlflow.tracking.MlflowClient()
    run_id = client.create_run(experiment_id=0).info.run_id

    from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import VarianceThreshold

    with mlflow.start_run(run_id=run_id):
        Normalizer().fit_transform(np.random.random((5, 5)))
        LabelEncoder().fit([1, 2, 2, 6])
        MinMaxScaler().fit_transform(50 * np.random.random((10, 10)))
        SimpleImputer().fit_transform([[1, 2], [np.nan, 3], [7, 6]])
        TfidfVectorizer().fit_transform(
            [
                "MLflow is an end-to-end machine learning platform.",
                "MLflow enables me to systematize my ML experimentation",
            ]
        )
        VarianceThreshold().fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    params, metrics, tags, artifacts = get_run_data(run_id)
    assert len(params) == 0
    assert len(metrics) == 0
    assert len(tags) == 0
    assert len(artifacts) == 0


@pytest.mark.large
def test_autolog_produces_expected_results_for_estimator_when_parent_also_defines_fit():
    """
    Test to prevent recurrences of https://github.com/mlflow/mlflow/issues/3574
    """
    mlflow.sklearn.autolog()

    # Construct two mock models - `ParentMod` and `ChildMod`, where ChildMod's fit() function
    # calls ParentMod().fit() and mutates a predefined, constant prediction value set by
    # ParentMod().fit(). We will then test that ChildMod.fit() completes and produces the
    # expected constant prediction value, guarding against regressions of
    # https://github.com/mlflow/mlflow/issues/3574 where ChildMod.fit() would either infinitely
    # recurse or yield the incorrect prediction result set by ParentMod.fit()

    class ParentMod(sklearn.base.BaseEstimator):
        def __init__(self):
            self.prediction = None

        def get_params(self, deep=False):
            return {}

        def fit(self, X, y):  # pylint: disable=unused-argument
            self.prediction = np.array([7])

        def predict(self, X):  # pylint: disable=unused-argument
            return self.prediction

    class ChildMod(ParentMod):
        def fit(self, X, y):
            super().fit(X, y)
            self.prediction = self.prediction + 1

    og_all_estimators = mlflow.sklearn.utils._all_estimators()
    new_all_estimators = og_all_estimators + [("ParentMod", ParentMod), ("ChildMod", ChildMod)]

    with mock.patch("mlflow.sklearn.utils._all_estimators", return_value=new_all_estimators):
        mlflow.sklearn.autolog()

    model = ChildMod()
    with mlflow.start_run() as run:
        model.fit(*get_iris())

    _, _, tags, _ = get_run_data(run.info.run_id)
    assert {"estimator_name": "ChildMod"}.items() <= tags.items()
    assert model.predict(1) == np.array([8])


def test_eval_and_log_metrics_for_regressor():
    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    # use simple `LinearRegression`, which only implements `fit`.
    model = sklearn.linear_model.LinearRegression()
    X, y_true = get_iris()
    X_eval = X[:-1, :]
    y_eval = y_true[:-1]
    with mlflow.start_run() as run:
        model = fit_model(model, X, y_true, "fake")
        eval_metrics = mlflow.sklearn.eval_and_log_metrics(
            model=model, X=X_eval, y_true=y_eval, prefix="eval_"
        )
    # Check correctness for the returned metrics/artifacts
    y_pred = model.predict(X_eval)
    assert eval_metrics == {
        "eval_score": model.score(X_eval, y_eval),
        "eval_mse": sklearn.metrics.mean_squared_error(y_eval, y_pred),
        "eval_rmse": np.sqrt(sklearn.metrics.mean_squared_error(y_eval, y_pred)),
        "eval_mae": sklearn.metrics.mean_absolute_error(y_eval, y_pred),
        "eval_r2_score": sklearn.metrics.r2_score(y_eval, y_pred),
    }

    # Check that logged metrics/artifacts are the same as returned by the method
    run_id = run.info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)
    assert metrics == eval_metrics
    assert len(artifacts) == 0


def test_eval_and_log_metrics_for_binary_classifier():
    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    import sklearn.ensemble

    # use RandomForestClassifier that has method [predict_proba], so that we can test
    # logging of (1) log_loss and (2) roc_auc_score.
    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)

    # use binary datasets to cover the test for roc curve & precision recall curve
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_eval = X[:-1, :]
    y_eval = y[:-1]

    with mlflow.start_run() as run:
        model = fit_model(model, X, y, "fit")
        eval_metrics = mlflow.sklearn.eval_and_log_metrics(
            model=model, X=X_eval, y_true=y_eval, prefix="val_"
        )

    y_pred = model.predict(X_eval)
    y_pred_prob = model.predict_proba(X_eval)
    # For binary classification, y_score only accepts the probability of greater label
    y_pred_prob_roc = y_pred_prob[:, 1]

    expected_metrics = {
        "val_score": model.score(X_eval, y_eval),
        "val_accuracy_score": sklearn.metrics.accuracy_score(y_eval, y_pred),
        "val_precision_score": sklearn.metrics.precision_score(y_eval, y_pred, average="weighted"),
        "val_recall_score": sklearn.metrics.recall_score(y_eval, y_pred, average="weighted"),
        "val_f1_score": sklearn.metrics.f1_score(y_eval, y_pred, average="weighted"),
        "val_log_loss": sklearn.metrics.log_loss(y_eval, y_pred_prob),
    }
    if _is_metric_supported("roc_auc_score"):
        expected_metrics["val_roc_auc_score"] = sklearn.metrics.roc_auc_score(
            y_eval, y_score=y_pred_prob_roc, average="weighted", multi_class="ovo",
        )
    assert eval_metrics == expected_metrics

    eval_artifacts = []
    if _is_plotting_supported():
        eval_artifacts.extend(
            [
                "{}.png".format("val_confusion_matrix"),
                "{}.png".format("val_roc_curve"),
                "{}.png".format("val_precision_recall_curve"),
            ]
        )

    # Check that logged artifacts/metrics are the same as the ones returned by the method
    run_id = run.info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)

    assert metrics == eval_metrics
    assert sorted(artifacts) == sorted(eval_artifacts)


def test_eval_and_log_metrics_matches_training_metrics():
    mlflow.sklearn.autolog()

    import sklearn.ensemble

    # use RandomForestClassifier that has method [predict_proba], so that we can test
    # logging of (1) log_loss and (2) roc_auc_score.
    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)

    # use binary datasets to cover the test for roc curve & precision recall curve
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_eval = X[:-1, :]
    y_eval = y[:-1]

    with mlflow.start_run() as run:
        model = fit_model(model, X, y, "fit")
        eval_metrics = mlflow.sklearn.eval_and_log_metrics(
            model=model, X=X_eval, y_true=y_eval, prefix="val_"
        )

    y_pred = model.predict(X_eval)
    y_pred_prob = model.predict_proba(X_eval)
    # For binary classification, y_score only accepts the probability of greater label
    y_pred_prob_roc = y_pred_prob[:, 1]

    expected_metrics = {
        "val_score": model.score(X_eval, y_eval),
        "val_accuracy_score": sklearn.metrics.accuracy_score(y_eval, y_pred),
        "val_precision_score": sklearn.metrics.precision_score(y_eval, y_pred, average="weighted"),
        "val_recall_score": sklearn.metrics.recall_score(y_eval, y_pred, average="weighted"),
        "val_f1_score": sklearn.metrics.f1_score(y_eval, y_pred, average="weighted"),
        "val_log_loss": sklearn.metrics.log_loss(y_eval, y_pred_prob),
    }
    if _is_metric_supported("roc_auc_score"):
        expected_metrics["val_roc_auc_score"] = sklearn.metrics.roc_auc_score(
            y_eval, y_score=y_pred_prob_roc, average="weighted", multi_class="ovo",
        )
    assert eval_metrics == expected_metrics

    eval_artifacts = []
    if _is_plotting_supported():
        eval_artifacts.extend(
            [
                "{}.png".format("val_confusion_matrix"),
                "{}.png".format("val_roc_curve"),
                "{}.png".format("val_precision_recall_curve"),
            ]
        )

    # Check that eval metrics/artifacts match the training metrics/artifacts
    run_id = run.info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)

    for key, value in eval_metrics.items():
        assert metrics[str(key)] == value
        assert metrics[str(key).replace("val_", "training_")] is not None

    for path in eval_artifacts:
        assert path in artifacts
        assert str(path).replace("val_", "training_") in artifacts


def test_eval_and_log_metrics_for_classifier_multi_class():
    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    import sklearn.ensemble

    # use RandomForestClassifier that has method [predict_proba], so that we can test
    # logging of (1) log_loss and (2) roc_auc_score.
    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)

    # use multi-class datasets to verify that roc curve & precision recall curve care not recorded
    X, y = get_iris()
    X_eval = X[:-1, :]
    y_eval = y[:-1]

    with mlflow.start_run() as run:
        model = fit_model(model, X, y, "fit")
        eval_metrics = mlflow.sklearn.eval_and_log_metrics(
            model=model, X=X_eval, y_true=y_eval, prefix="eval_"
        )

    # Check the contents of the returned artifacts and metrics
    y_pred = model.predict(X_eval)
    y_pred_prob = model.predict_proba(X_eval)

    expected_metrics = {
        "eval_score": model.score(X_eval, y_eval),
        "eval_accuracy_score": sklearn.metrics.accuracy_score(y_eval, y_pred),
        "eval_precision_score": sklearn.metrics.precision_score(y_eval, y_pred, average="weighted"),
        "eval_recall_score": sklearn.metrics.recall_score(y_eval, y_pred, average="weighted"),
        "eval_f1_score": sklearn.metrics.f1_score(y_eval, y_pred, average="weighted"),
        "eval_log_loss": sklearn.metrics.log_loss(y_eval, y_pred_prob),
    }
    if _is_metric_supported("roc_auc_score"):
        expected_metrics["eval_roc_auc_score"] = sklearn.metrics.roc_auc_score(
            y_eval, y_score=y_pred_prob, average="weighted", multi_class="ovo",
        )

    assert eval_metrics == expected_metrics

    eval_artifacts = []
    if _is_plotting_supported():
        eval_artifacts = ["{}.png".format("eval_confusion_matrix")]

    # Check that the logged metrics/artifacts are the same as the ones returned by the method.
    run_id = run.info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)

    assert metrics == eval_metrics
    assert artifacts == eval_artifacts


def test_eval_and_log_metrics_with_estimator(fit_func_name):
    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    import sklearn.cluster

    # use `KMeans` because it implements `fit`, `fit_transform`, and `fit_predict`.
    model = sklearn.cluster.KMeans()
    X, y = get_iris()
    X_eval = X[:-1, :]
    y_eval = y[:-1]

    with mlflow.start_run() as run:
        model = fit_model(model, X, y, fit_func_name)
        eval_metrics = mlflow.sklearn.eval_and_log_metrics(
            model=model, X=X_eval, y_true=y_eval, prefix="eval_"
        )

    # Check contents of returned artifacts/metrics
    assert eval_metrics == {"eval_score": model.score(X_eval, y_eval)}

    # Check that the logged metrics are the same as returned by the method.
    run_id = run.info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)

    assert metrics == eval_metrics
    assert len(artifacts) == 0


def test_eval_and_log_metrics_with_meta_estimator():
    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    import sklearn.preprocessing
    import sklearn.svm

    estimators = [
        ("std_scaler", sklearn.preprocessing.StandardScaler()),
        ("svc", sklearn.svm.SVC()),
    ]
    model = sklearn.pipeline.Pipeline(estimators)
    X, y = get_iris()
    X_eval = X[:-1, :]
    y_eval = y[:-1]

    with mlflow.start_run() as run:
        model.fit(X, y)
        eval_metrics = mlflow.sklearn.eval_and_log_metrics(
            model=model, X=X_eval, y_true=y_eval, prefix="eval_"
        )

    eval_artifacts = ["{}.png".format("eval_confusion_matrix")] if _is_plotting_supported() else []

    # Check that the logged metrics/artifacts for the run are exactly those returned by the call
    run_id = run.info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)
    assert sorted(artifacts) == sorted(eval_artifacts)
    assert metrics == eval_metrics

    # Check the actual metrics and artifacts
    y_pred = model.predict(X_eval)

    # SVC does not support probability predictions so the corresponding metrics (log_loss, auc)
    # are missing.
    expected_metrics = {
        "eval_score": model.score(X_eval, y_eval),
        "eval_accuracy_score": sklearn.metrics.accuracy_score(y_eval, y_pred),
        "eval_precision_score": sklearn.metrics.precision_score(y_eval, y_pred, average="weighted"),
        "eval_recall_score": sklearn.metrics.recall_score(y_eval, y_pred, average="weighted"),
        "eval_f1_score": sklearn.metrics.f1_score(y_eval, y_pred, average="weighted"),
    }
    assert eval_metrics == expected_metrics


def test_eval_and_log_metrics_with_new_run():
    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    # use simple `LinearRegression`, which only implements `fit`.
    model = sklearn.linear_model.LinearRegression()
    X, y_true = get_iris()
    X_eval = X[:-1, :]
    y_eval = y_true[:-1]
    model = fit_model(model, X, y_true, "fake")

    eval_metrics = mlflow.sklearn.eval_and_log_metrics(
        model=model, X=X_eval, y_true=y_eval, prefix="eval_"
    )
    # Check the contents for the metrics and artifacts
    y_pred = model.predict(X_eval)
    assert eval_metrics == {
        "eval_score": model.score(X_eval, y_eval),
        "eval_mse": sklearn.metrics.mean_squared_error(y_eval, y_pred),
        "eval_rmse": np.sqrt(sklearn.metrics.mean_squared_error(y_eval, y_pred)),
        "eval_mae": sklearn.metrics.mean_absolute_error(y_eval, y_pred),
        "eval_r2_score": sklearn.metrics.r2_score(y_eval, y_pred),
    }

    # Check the the logged metrics/artifacts are the same as the returned ones.
    assert mlflow.active_run() is not None
    run_id = mlflow.active_run().info.run_id
    _, metrics, _, artifacts = get_run_data(run_id)
    assert eval_metrics == metrics
    assert len(artifacts) == 0
    mlflow.end_run()


def test_eval_and_log_metrics_with_noscore_estimator():
    from sklearn.base import BaseEstimator

    # disable autologging so that we can check for the sole existence of eval-time metrics
    mlflow.sklearn.autolog(disable=True)

    # Define a fake estimator that can do predictions but does not support 'score'
    class FakeEstimator(BaseEstimator):
        def predict(self, X):
            return np.random.random(np.shape(X)[-1])

    # use simple `LinearRegression`, which only implements `fit`.
    model = FakeEstimator()
    X_eval, y_eval = get_iris()

    eval_metrics = mlflow.sklearn.eval_and_log_metrics(
        model=model, X=X_eval, y_true=y_eval, prefix="eval_"
    )
    _, metrics, _, artifacts = get_run_data(mlflow.active_run().info.run_id)

    mlflow.end_run()

    # No artifacts should be generated
    assert len(metrics) == 0
    assert len(eval_metrics) == 0
    assert len(artifacts) == 0


def test_eval_and_log_metrics_throws_with_invalid_args():
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import SpectralClustering

    X, y_true = get_iris()
    model = LinearRegression()

    with pytest.raises(ValueError, match="Must specify a non-empty prefix"):
        mlflow.sklearn.eval_and_log_metrics(model=model, X=X, y_true=y_true, prefix="")

    with pytest.raises(ValueError, match="Must specify a non-empty prefix"):
        mlflow.sklearn.eval_and_log_metrics(model=model, X=X, y_true=y_true, prefix=None)

    with pytest.raises(ValueError, match="not a sklearn estimator"):
        mlflow.sklearn.eval_and_log_metrics(model={}, X=X, y_true=y_true, prefix="val_")

    with pytest.raises(ValueError, match="Model does not support predictions"):
        mlflow.sklearn.eval_and_log_metrics(
            model=SpectralClustering(), X=X, y_true=y_true, prefix="val_"
        )


def test_metric_computation_handles_absent_labels():
    """
    Verifies that autologging metric computation does not fail For models that do not require
    labels as inputs to training, such as clustering models and other unsupervised techniques.
    """
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()

    with mlflow.start_run() as run:
        # Train a clustering model solely on samples, without specifying labels
        model.fit(X=get_iris()[0])

    params, metrics, tags, artifacts = get_run_data(run.info.run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    # We expect metrics to be absent because labels are required to compute autologging metrics
    # for sklearn models
    assert not metrics
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
