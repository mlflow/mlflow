import contextlib
import doctest
import functools
import inspect
import json
import os
import pickle
import re
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.base
import sklearn.cluster
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
from packaging.version import Version
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import uniform

import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.sklearn.utils import (
    _get_arg_names,
    _is_estimator_html_repr_supported,
    _is_metric_supported,
    _is_plotting_supported,
    _log_child_runs_info,
    _log_estimator_content,
)
from mlflow.types.utils import _infer_schema
from mlflow.utils import _truncate_dict
from mlflow.utils.autologging_utils import MlflowAutologgingQueueingClient
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
from mlflow.utils.validation import (
    MAX_ENTITY_KEY_LENGTH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
)

FIT_FUNC_NAMES = ["fit", "fit_transform", "fit_predict"]
TRAINING_SCORE = "training_score"
ESTIMATOR_CLASS = "estimator_class"
ESTIMATOR_NAME = "estimator_name"
MODEL_DIR = "model"


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
    return MlflowClient().get_run(run_id)


def get_run_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    return data.params, data.metrics, tags, artifacts


def load_model_by_run_id(run_id):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_DIR}")


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
    ("max_tuning_runs", "total_runs", "output_statement"),
    [
        (0, 4, "Logging no runs, all will be omitted"),
        (0, 1, "Logging no runs, one run will be omitted"),
        (1, 1, "Logging the best run, no runs will be omitted"),
        (5, 4, "Logging all runs, no runs will be omitted"),
        (4, 4, "Logging all runs, no runs will be omitted"),
        (2, 5, "Logging the 2 best runs, 3 runs will be omitted"),
    ],
)
def test_autolog_max_tuning_runs_logs_info_correctly(max_tuning_runs, total_runs, output_statement):
    with mock.patch("mlflow.sklearn.utils._logger.info") as mock_info:
        _log_child_runs_info(max_tuning_runs, total_runs)
        mock_info.assert_called_once()
        mock_info.called_once_with(output_statement)


def test_autolog_does_not_terminate_active_run():
    mlflow.sklearn.autolog()
    mlflow.start_run()
    sklearn.cluster.KMeans().fit(*get_iris())
    assert mlflow.active_run() is not None
    mlflow.end_run()


def test_extra_tags_sklearn_autolog():
    mlflow.sklearn.autolog(extra_tags={"test_tag": "sklearn_autolog"})
    sklearn.cluster.KMeans().fit(*get_iris())
    assert mlflow.active_run() is None

    run = mlflow.last_active_run()
    assert run.data.tags["test_tag"] == "sklearn_autolog"
    assert run.data.tags[mlflow.utils.mlflow_tags.MLFLOW_AUTOLOGGING] == "sklearn"


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
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
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
        expected_metrics["training_roc_auc"] = sklearn.metrics.roc_auc_score(
            y_true,
            y_score=y_pred_prob_roc,
            average="weighted",
            multi_class="ovo",
        )

    assert metrics == expected_metrics

    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts

    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    plot_names = []
    if _is_plotting_supported():
        plot_names.extend(
            [
                "training_confusion_matrix.png",
                "training_roc_curve.png",
                "training_precision_recall_curve.png",
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
        expected_metrics["training_roc_auc"] = sklearn.metrics.roc_auc_score(
            y_true,
            y_score=y_pred_prob,
            average="weighted",
            multi_class="ovo",
        )

    assert metrics == expected_metrics

    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts

    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    plot_names = []
    if _is_plotting_supported():
        plot_names = ["training_confusion_matrix.png"]

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
        "training_mean_squared_error": sklearn.metrics.mean_squared_error(y_true, y_pred),
        "training_root_mean_squared_error": np.sqrt(
            sklearn.metrics.mean_squared_error(y_true, y_pred)
        ),
        "training_mean_absolute_error": sklearn.metrics.mean_absolute_error(y_true, y_pred),
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
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def disable_validate_params(cls):
    return (
        mock.patch(f"{cls}._validate_params")
        if Version(sklearn.__version__) >= Version("1.2.dev0")
        else contextlib.nullcontext()
    )


def test_get_params_returns_dict_that_has_more_keys_than_max_params_tags_per_batch():
    mlflow.sklearn.autolog()

    large_params = {str(i): str(i) for i in range(MAX_PARAMS_TAGS_PER_BATCH + 1)}
    X, y = get_iris()

    with disable_validate_params("sklearn.cluster.KMeans"), mock.patch(
        "sklearn.cluster.KMeans.get_params", return_value=large_params
    ):
        with mlflow.start_run() as run:
            model = sklearn.cluster.KMeans()
            model.fit(X, y)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run.info.run_id)
    assert params == large_params
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert_predict_equal(loaded_model, model, X)


@pytest.mark.parametrize(
    ("long_params", "messages"),
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

    with disable_validate_params("sklearn.cluster.KMeans"), mock.patch(
        "sklearn.cluster.KMeans.get_params", return_value=long_params
    ), mock.patch("mlflow.utils._logger.warning") as mock_warning, mlflow.start_run() as run:
        model = sklearn.cluster.KMeans()
        model.fit(X, y)

    for idx, msg in enumerate(messages):
        assert mock_warning.call_args_list[idx].startswith(msg)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run.info.run_id)
    assert params == truncate_dict(long_params)
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
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
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
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
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(SGDRegressor.score) == inspect.signature(mock_score)

    SGDRegressor.score = mock_score
    model = SGDRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y, intercept_init=0)

    assert len(mock_obj.call_args_list) == 1
    mock_call_args = mock_obj.call_args_list[0][0]
    assert len(mock_call_args) == 3
    np.testing.assert_array_equal(mock_call_args[0], X)
    np.testing.assert_array_equal(mock_call_args[1], y)
    assert mock_call_args[2] is None

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
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

    assert len(mock_obj.call_args_list) == 1
    mock_call_args = mock_obj.call_args_list[0][0]
    assert len(mock_call_args) == 3
    np.testing.assert_array_equal(mock_call_args[0], X)
    np.testing.assert_array_equal(mock_call_args[1], y)
    np.testing.assert_array_equal(mock_call_args[2], sample_weight)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_only_fit_contains_sample_weight():
    mlflow.sklearn.autolog()

    from sklearn.linear_model import RANSACRegressor

    assert "sample_weight" in _get_arg_names(RANSACRegressor.fit)
    assert "sample_weight" not in _get_arg_names(RANSACRegressor.score)

    mock_obj = mock.Mock()

    def mock_score(self, X, y, **kwargs):
        mock_obj(X, y)
        return 0

    RANSACRegressor.score = mock_score
    model = RANSACRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)

    assert len(mock_obj.call_args_list) == 1
    mock_call_args = mock_obj.call_args_list[0][0]
    assert len(mock_call_args) == 2
    np.testing.assert_array_equal(mock_call_args[0], X)
    np.testing.assert_array_equal(mock_call_args[1], y)

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
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
        mock_obj(X, y, sample_weight)
        return 0

    assert inspect.signature(GaussianProcessRegressor.score) == inspect.signature(mock_score)

    GaussianProcessRegressor.score = mock_score
    model = GaussianProcessRegressor()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)

    assert len(mock_obj.call_args_list) == 1
    mock_call_args = mock_obj.call_args_list[0][0]
    assert len(mock_call_args) == 3
    np.testing.assert_array_equal(mock_call_args[0], X)
    np.testing.assert_array_equal(mock_call_args[1], y)
    assert mock_call_args[2] is None

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(stringify_dict_values(model.get_params(deep=True)))
    assert {TRAINING_SCORE: pytest.approx(model.score(X, y), abs=1e-6)}.items() <= metrics.items()
    assert tags == get_expected_class_tags(model)
    assert MODEL_DIR in artifacts
    assert_predict_equal(load_model_by_run_id(run_id), model, X)


def test_autolog_terminates_run_when_active_run_does_not_exist_and_fit_fails():
    mlflow.sklearn.autolog()

    with pytest.raises(
        ValueError,
        match=r"(Penalty term must be positive|The 'C' parameter of LinearSVC must be a float)",
    ):
        sklearn.svm.LinearSVC(C=-1).fit(*get_iris())

    latest_run = mlflow.search_runs().iloc[0]
    assert mlflow.active_run() is None
    assert latest_run.status == "FAILED"


def test_autolog_does_not_terminate_run_when_active_run_exists_and_fit_fails():
    mlflow.sklearn.autolog()
    run = mlflow.start_run()

    with pytest.raises(
        ValueError,
        match=r"(Penalty term must be positive|The 'C' parameter of LinearSVC must be a float)",
    ):
        sklearn.svm.LinearSVC(C=-1).fit(*get_iris())

    assert mlflow.active_run() is not None
    assert mlflow.active_run() is run
    mlflow.end_run()


def test_autolog_emits_warning_message_when_score_fails():
    mlflow.sklearn.autolog()

    model = sklearn.cluster.KMeans()

    @functools.wraps(model.score)
    def throwing_score(X, y=None, sample_weight=None):
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
    def throwing_metrics(y_true, y_pred):
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
        f"score_{i}": sklearn.metrics.make_scorer(lambda y, y_pred, **kwargs: 10)
        for i in range(metrics_size)
    }

    with mlflow.start_run(), mock.patch("mlflow.sklearn.utils._logger.warning") as mock_warning:
        svc = sklearn.svm.SVC()
        cv_model = sklearn.model_selection.GridSearchCV(
            svc, {"C": [1]}, n_jobs=1, scoring=metrics_to_log, refit=False
        )
        cv_model.fit(*get_iris())

        # Ensure `cv_model.predict` fails with `NotFittedError` or `AttributeError`
        err = (
            NotFittedError if Version(sklearn.__version__) <= Version("0.24.2") else AttributeError
        )
        match = r"GridSearchCV.+predict"
        with pytest.raises(err, match=match):
            cv_model.predict([[0, 0, 0, 0]])

        # Count how many times `mock_warning` has been called on not-fitted `predict` failure
        call_count = len(
            [args for args in mock_warning.call_args_list if re.search(match, args[0][0])]
        )
        # If `_is_plotting_supported` returns True (meaning sklearn version is >= 0.22.0),
        # `mock_warning` should have been called twice, once for metrics, once for artifacts.
        # Otherwise, only once for metrics.
        call_count_expected = 2 if mlflow.sklearn.utils._is_plotting_supported() else 1
        assert call_count == call_count_expected


@pytest.mark.parametrize(
    ("cv_class", "search_space"),
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
        log_input_examples=True,
        log_model_signatures=True,
        max_tuning_runs=max_tuning_runs,
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
            f"best_{param_name}": str(param_value)
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

    best_estimator = mlflow.sklearn.load_model(f"runs:/{run_id}/best_estimator")
    assert isinstance(best_estimator, sklearn.svm.SVC)
    cv_model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_DIR}")
    assert isinstance(cv_model, cv_class)

    # Ensure that a signature and input example are produced for the best estimator
    best_estimator_conf = get_model_conf(run.info.artifact_uri, "best_estimator")
    assert best_estimator_conf.signature == infer_signature(X, best_estimator.predict(X[:5]))

    best_estimator_path = os.path.join(run.info.artifact_uri, "best_estimator")
    input_example = _read_example(best_estimator_conf, best_estimator_path)
    best_estimator.predict(input_example)  # Ensure that input example evaluation succeeds

    client = MlflowClient()
    child_runs = client.search_runs(
        run.info.experiment_id, f"tags.`mlflow.parentRunId` = '{run_id}'"
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
            [f"params.`{key}` = '{value}'" for key, value in result_params.items()]
        )
        search_filter = f"tags.`mlflow.parentRunId` = '{run_id}' and {params_search_clause}"
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
            [f"params.`{key}` = '{value}'" for key, value in result_params.items()]
        )
        search_filter = f"tags.`mlflow.parentRunId` = '{run_id}' and {params_search_clause}"
        child_runs = client.search_runs(run.info.experiment_id, search_filter)
        assert len(child_runs) == 0


def test_parameter_search_handles_large_volume_of_metric_outputs():
    mlflow.sklearn.autolog()

    metrics_size = MAX_METRICS_PER_BATCH + 10
    metrics_to_log = {
        f"score_{i}": sklearn.metrics.make_scorer(lambda y, y_pred, **kwargs: 10)
        for i in range(metrics_size)
    }

    with mlflow.start_run() as run:
        svc = sklearn.svm.SVC()
        cv_model = sklearn.model_selection.GridSearchCV(
            svc, {"C": [1]}, n_jobs=1, scoring=metrics_to_log, refit=False
        )
        cv_model.fit(*get_iris())
        run_id = run.info.run_id

    client = MlflowClient()
    child_runs = client.search_runs(
        run.info.experiment_id, f"tags.`mlflow.parentRunId` = '{run_id}'"
    )
    assert len(child_runs) == 1
    child_run = child_runs[0]

    assert len(child_run.data.metrics) >= metrics_size


@pytest.mark.parametrize("data_type", [pd.DataFrame, np.array, csr_matrix, csc_matrix])
def test_autolog_logs_signature_and_input_example(data_type):
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    X, y = get_iris()
    X = data_type(X)
    y = np.array(y) if data_type in [csr_matrix, csc_matrix] else data_type(y)
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


def test_autolog_metrics_input_example_and_signature_do_not_reflect_training_mutations():
    from sklearn.base import BaseEstimator, TransformerMixin

    X_train = pd.DataFrame(
        {
            "Total Volume": [64236.62, 54876.98, 118220.22],
            "Total Bags": [8696.87, 9505.56, 8145.35],
            "Small Bags": [8603.62, 9408.07, 8042.21],
            "Large Bags": [93.25, 97.49, 103.14],
            "XLarge Bags": [0.0, 0.0, 0.0],
        }
    )
    y_train = pd.Series([1.33, 1.35, 0.93])

    class CustomTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            # Perform arbitary transformation
            if "XXLarge Bags" in X.columns:
                raise Exception("Found unexpected 'XXLarge Bags' column!")
            X["XXLarge Bags"] = X["XLarge Bags"] + 1
            return X

    mlflow.sklearn.autolog(log_models=True, log_model_signatures=True, log_input_examples=True)

    sk_pipeline = sklearn.pipeline.make_pipeline(
        CustomTransformer(), sklearn.linear_model.LinearRegression()
    )
    sk_pipeline.fit(X_train, y_train)

    run_artifact_uri = mlflow.last_active_run().info.artifact_uri
    model_conf = get_model_conf(run_artifact_uri)
    input_example = pd.read_json(
        os.path.join(run_artifact_uri, "model", "input_example.json"), orient="split"
    )
    model_signature_input_names = [inp.name for inp in model_conf.signature.inputs.inputs]
    assert "XLarge Bags" in model_signature_input_names
    assert "XLarge Bags" in input_example.columns
    assert "XXLarge Bags" not in model_signature_input_names
    assert "XXLarge Bags" not in input_example.columns

    metrics = get_run_data(mlflow.last_active_run().info.run_id)[1]
    assert "training_r2_score" in metrics
    assert "training_root_mean_squared_error" in metrics


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
        throwing_X[:5]

    mlflow.sklearn.autolog()
    model = sklearn.linear_model.LinearRegression()

    with mlflow.start_run() as run, mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        model.fit(throwing_X, y)

    model_conf = get_model_conf(run.info.artifact_uri)

    assert mock_warning.call_count == 2
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

    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    # Note that `mock_warning` will be called twice because if `predict` throws, `score` also throws
    with mlflow.start_run() as run, mock.patch(
        "sklearn.linear_model.LinearRegression.predict", side_effect=Exception("Failed")
    ), mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
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


def test_autolog_does_not_warn_when_model_has_transform_function():
    X, y = get_iris()

    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
    with mlflow.start_run() as run, mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        estimators = [
            ("std_scaler", sklearn.preprocessing.StandardScaler()),
        ]
        model = sklearn.pipeline.Pipeline(estimators)
        model.fit(X, y)

    # Warning not called
    msg = "Failed to infer model signature:"
    assert all(msg not in c[0] for c in mock_warning.call_args_list)

    model_conf = get_model_conf(run.info.artifact_uri)
    assert "signature" in model_conf.to_dict()


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


@pytest.mark.parametrize("log_datasets", [True, False])
def test_sklearn_autolog_log_datasets_configuration(log_datasets):
    X, y = get_iris()

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog(log_datasets=log_datasets)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

    run_id = run.info.run_id
    client = MlflowClient()
    dataset_inputs = client.get_run(run_id).inputs.dataset_inputs
    if log_datasets:
        assert len(dataset_inputs) == 1
        feature_schema = _infer_schema(X)
        target_schema = _infer_schema(y)
        assert dataset_inputs[0].dataset.schema == json.dumps(
            {
                "mlflow_tensorspec": {
                    "features": feature_schema.to_json(),
                    "targets": target_schema.to_json(),
                }
            }
        )
    else:
        assert len(dataset_inputs) == 0


def test_sklearn_autolog_log_datasets_with_predict():
    X, y = get_iris()

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog(log_datasets=True)
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)
        model.predict(X)

    run_id = run.info.run_id
    client = MlflowClient()
    dataset_inputs = client.get_run(run_id).inputs.dataset_inputs

    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    feature_schema = _infer_schema(X)
    target_schema = _infer_schema(y)
    assert dataset_inputs[0].dataset.schema == json.dumps(
        {
            "mlflow_tensorspec": {
                "features": feature_schema.to_json(),
                "targets": target_schema.to_json(),
            }
        }
    )
    assert dataset_inputs[1].tags[0].value == "eval"
    assert dataset_inputs[1].dataset.schema == json.dumps(
        {
            "mlflow_tensorspec": {
                "features": feature_schema.to_json(),
                "targets": None,
            }
        }
    )


def test_sklearn_autolog_log_datasets_without_explicit_run():
    X, y = get_iris()

    mlflow.sklearn.autolog(log_datasets=True)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    model.predict(X)

    run_id = getattr(model, "_mlflow_run_id")
    client = MlflowClient()
    dataset_inputs = client.get_run(run_id).inputs.dataset_inputs

    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    feature_schema = _infer_schema(X)
    target_schema = _infer_schema(y)
    assert dataset_inputs[0].dataset.schema == json.dumps(
        {
            "mlflow_tensorspec": {
                "features": feature_schema.to_json(),
                "targets": target_schema.to_json(),
            }
        }
    )
    assert dataset_inputs[1].tags[0].value == "eval"
    assert dataset_inputs[1].dataset.schema == json.dumps(
        {
            "mlflow_tensorspec": {
                "features": feature_schema.to_json(),
                "targets": None,
            }
        }
    )


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
    client = MlflowClient()
    run_id = client.create_run(experiment_id=0).info.run_id

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer

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
        ColumnTransformer([("norm", Normalizer(), [0])]).fit_transform([[0]])

    params, metrics, tags, artifacts = get_run_data(run_id)
    assert len(params) == 0
    assert len(metrics) == 0
    assert len(tags) == 0
    assert len(artifacts) == 0


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

        def fit(self, X, y):
            self.prediction = np.array([7])

        def predict(self, X):
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


@pytest.mark.parametrize("cross_val_func_name", mlflow.sklearn._apis_autologging_disabled)
def test_autolog_disabled_on_sklearn_cross_val_api(cross_val_func_name):
    mlflow.sklearn.autolog()
    from sklearn import linear_model

    def assert_autolog_disabled_during_exec_cross_val_fun(run_):
        params, metrics, tags, artifacts = get_run_data(run_.info.run_id)
        assert params == {}
        assert metrics == {}
        assert tags == {}
        assert artifacts == []

    diabetes = sklearn.datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()

    if cross_val_func_name == "validation_curve":
        extra_params = {"param_name": "max_iter", "param_range": [10, 100]}
    else:
        extra_params = {}

    cross_val_func = getattr(sklearn.model_selection, cross_val_func_name)
    with mlflow.start_run() as run:
        cross_val_func(lasso, X, y, cv=3, **extra_params)
        assert_autolog_disabled_during_exec_cross_val_fun(run)

    # Ensure cross_val_func doesn't start a new run
    exp_id = mlflow.tracking.fluent._get_experiment_id()
    runs_before = mlflow.search_runs([exp_id])
    cross_val_func(lasso, X, y, cv=3, **extra_params)
    runs_after = mlflow.search_runs([exp_id])
    assert len(runs_before) == len(runs_after)


def load_json_artifact(artifact_path):
    fpath = mlflow.get_artifact_uri(artifact_path).replace("file://", "")
    with open(fpath) as f:
        return json.load(f)


def test_basic_post_training_metric_autologging():
    from sklearn import metrics as sklmetrics

    mlflow.sklearn.autolog()

    model = sklearn.linear_model.LogisticRegression(solver="saga", max_iter=100, random_state=0)
    X, y = get_iris()

    with mlflow.start_run() as run:
        model.fit(X, y)

        eval1_X, eval1_y = X[0::3], y[0::3]
        eval2_X, eval2_y = X[1::3], y[1::3]

        pred1_y = model.predict(X=eval1_X)
        pred2_y = model.predict(eval2_X)

        r2_score_data1 = sklmetrics.r2_score(eval1_y, pred1_y)
        recall_score_data1 = sklmetrics.recall_score(eval1_y, pred1_y, average="macro")
        r2_score_data2 = sklmetrics.r2_score(eval2_y, pred2_y)
        lor_score_data1 = model.score(eval1_X, eval1_y)
        recall_score2_data2 = sklmetrics.recall_score(eval2_y, pred2_y, average="micro")

        scorer1 = sklmetrics.make_scorer(sklmetrics.recall_score, average="micro")
        recall_score3_data2 = scorer1(model, eval2_X, eval2_y)

        scorer2 = sklmetrics.make_scorer(sklmetrics.recall_score, average="macro")
        recall_score4_data2 = scorer2(model, eval2_X, eval2_y)

        eval1_X, eval1_y = eval1_X.copy(), eval1_y.copy()
        # In metric key, it will include dataset name as "eval1_X-2"
        lor_score_data1_2 = model.score(eval1_X, eval1_y)

        # In metric key, it will include dataset name as "unknown_dataset"
        lor_score_data1_3 = model.score(eval1_X.copy(), eval1_y.copy())

        metric_info = load_json_artifact("metric_info.json")

    run_id = run.info.run_id
    _, metrics, _, _ = get_run_data(run_id)
    post_training_metrics = {k: v for k, v in metrics.items() if not k.startswith("training_")}

    assert post_training_metrics == {
        "r2_score_eval1_X": r2_score_data1,
        "recall_score_eval1_X": recall_score_data1,
        "r2_score-2_eval2_X": r2_score_data2,
        "LogisticRegression_score_eval1_X": lor_score_data1,
        "recall_score-2_eval2_X": recall_score2_data2,
        "recall_score-3_eval2_X": recall_score3_data2,
        "recall_score-4_eval2_X": recall_score4_data2,
        "LogisticRegression_score-2_eval1_X-2": lor_score_data1_2,
        "LogisticRegression_score-3_unknown_dataset": lor_score_data1_3,
    }

    lor_score_3_cmd = "LogisticRegression.score(X=<ndarray>, y=<ndarray>)"
    recall_score4_eval2_X_cmd = "recall_score(y_true=eval2_y, y_pred=y_pred, average='macro')"
    assert metric_info == {
        "LogisticRegression_score-2_eval1_X-2": "LogisticRegression.score(X=eval1_X, y=eval1_y)",
        "LogisticRegression_score-3_unknown_dataset": lor_score_3_cmd,
        "LogisticRegression_score_eval1_X": "LogisticRegression.score(X=eval1_X, y=eval1_y)",
        "r2_score-2_eval2_X": "r2_score(y_true=eval2_y, y_pred=pred2_y)",
        "r2_score_eval1_X": "r2_score(y_true=eval1_y, y_pred=pred1_y)",
        "recall_score-2_eval2_X": "recall_score(y_true=eval2_y, y_pred=pred2_y, average='micro')",
        "recall_score-3_eval2_X": "recall_score(y_true=eval2_y, y_pred=y_pred, average='micro')",
        "recall_score-4_eval2_X": recall_score4_eval2_X_cmd,
        "recall_score_eval1_X": "recall_score(y_true=eval1_y, y_pred=pred1_y, average='macro')",
    }

    mlflow.sklearn.autolog(disable=True)

    # Test patched methods generate the same results with unpatched methods.
    recall_score_data1_original = sklmetrics.recall_score(eval1_y, pred1_y, average="macro")
    assert np.isclose(recall_score_data1_original, recall_score_data1)

    lor_score_data1_original = model.score(eval1_X, eval1_y)
    assert np.isclose(lor_score_data1_original, lor_score_data1)

    pred1_y_original = model.predict(eval1_X)
    np.testing.assert_allclose(pred1_y_original, pred1_y)


@pytest.mark.parametrize("metric_name", mlflow.sklearn._get_metric_name_list())
def test_run_metric_api_doc_example(metric_name):
    from sklearn import metrics

    mlflow.sklearn.autolog()
    metric_api = getattr(metrics, metric_name)
    doctest.run_docstring_examples(metric_api.__doc__, {}, verbose=True)


def test_post_training_metric_autologging_for_predict_prob():
    import sklearn.linear_model

    mlflow.sklearn.autolog()
    from sklearn.metrics import roc_auc_score

    X, y = get_iris()
    lor_model = sklearn.linear_model.LogisticRegression(solver="saga", max_iter=100, random_state=0)
    with mlflow.start_run() as run:
        lor_model.fit(X, y)
        y_prob = lor_model.predict_proba(X)
        y_true_onehot = np.eye(3)[y]
        roc_auc_metric = roc_auc_score(y_true_onehot, y_prob)

    _, metrics, _, _ = get_run_data(run.info.run_id)
    assert metrics["roc_auc_score_X"] == roc_auc_metric


def test_post_training_metric_autologging_patch_transform():
    mlflow.sklearn.autolog()
    X, y = get_iris()
    kmeans_model = sklearn.cluster.KMeans().fit(X, y)
    with mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.register_prediction_input_dataset"
    ) as mock_register_prediction_input_dataset:
        kmeans_model.transform(X)
        mock_register_prediction_input_dataset.assert_called_once()


def test_nested_metric_call_is_disabled():
    mlflow.sklearn.autolog()

    X, y = get_iris()
    eval1_X, eval1_y = X[0::3], y[0::3]
    lr_model = sklearn.linear_model.LinearRegression()

    with mlflow.start_run():
        with mock.patch(
            "mlflow.sklearn._AutologgingMetricsManager.log_post_training_metric"
        ) as patched_log_post_training_metric:
            # test post training metric logging disabled in fit scope
            lr_model.fit(X, y)
            patched_log_post_training_metric.assert_not_called()

            patched_log_post_training_metric.reset_mock()
            # test post training metric logging called only once in model.score
            lr_model.score(eval1_X, eval1_y)
            assert patched_log_post_training_metric.call_count == 1
            assert (
                patched_log_post_training_metric.call_args[0][1] == "LinearRegression_score_eval1_X"
            )


def test_multi_model_interleaved_fit_and_post_train_metric_call():
    mlflow.sklearn.autolog()
    from sklearn.metrics import mean_squared_error

    X, y = get_iris()
    eval1_X, eval1_y = X[0::3], y[0::3]
    eval2_X, eval2_y = X[1::3], y[1::3]

    lr_model1 = sklearn.linear_model.LinearRegression(fit_intercept=True)
    lr_model2 = sklearn.linear_model.LinearRegression(fit_intercept=False)

    with mlflow.start_run() as run1:
        lr_model1.fit(X, y)

    with mlflow.start_run() as run2:
        lr_model2.fit(X, y)

    model1_r2_score = lr_model1.score(eval1_X, eval1_y)
    model2_r2_score = lr_model2.score(eval2_X, eval2_y)

    pred1_y = lr_model1.predict(eval1_X)
    model1_mse = mean_squared_error(eval1_y, pred1_y)

    pred2_y = lr_model2.predict(eval2_X)
    model2_mse = mean_squared_error(eval2_y, pred2_y)

    _, metrics1, _, _ = get_run_data(run1.info.run_id)
    assert metrics1["LinearRegression_score_eval1_X"] == model1_r2_score
    assert metrics1["mean_squared_error_eval1_X"] == model1_mse

    _, metrics2, _, _ = get_run_data(run2.info.run_id)
    assert metrics2["LinearRegression_score_eval2_X"] == model2_r2_score
    assert metrics2["mean_squared_error_eval2_X"] == model2_mse


@pytest.mark.parametrize(
    "scoring", [None, sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)]
)
def test_meta_estimator_disable_nested_post_training_autologging(scoring):
    import sklearn.metrics
    import sklearn.svm

    mlflow.sklearn.autolog()

    X, y = get_iris()
    with mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.register_model"
    ) as mock_register_model, mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.is_metric_value_loggable"
    ) as mock_is_metric_value_loggable, mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.log_post_training_metric"
    ) as mock_log_post_training_metric, mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.register_prediction_input_dataset"
    ) as mock_register_prediction_input_dataset:
        with mlflow.start_run():
            svc = sklearn.svm.SVC()
            cv_model = sklearn.model_selection.GridSearchCV(
                svc, {"C": [1, 0.5]}, n_jobs=1, scoring=scoring
            )
            cv_model.fit(X, y)
            cv_model.predict(X)
            cv_model.score(X, y)
            mock_register_model.assert_called_once()
            assert mock_is_metric_value_loggable.call_count <= 1
            assert mock_log_post_training_metric.call_count <= 1
            assert mock_register_prediction_input_dataset.call_count <= 1


@pytest.mark.parametrize(
    "scoring", [None, sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)]
)
def test_meta_estimator_post_training_autologging(scoring):
    X, y = get_iris()
    eval1_X, eval1_y = X[0::3], y[0::3]

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        lor = sklearn.linear_model.LogisticRegression(solver="saga", random_state=0)
        cv_model = sklearn.model_selection.GridSearchCV(
            lor, {"max_iter": [5, 10, 15]}, n_jobs=1, scoring=scoring
        )
        cv_model.fit(X, y)
        pred1_y = cv_model.predict(eval1_X)
        accuracy_score = sklearn.metrics.accuracy_score(eval1_y, pred1_y, normalize=False)
        cv_score = cv_model.score(eval1_X, eval1_y)

        _, metrics, _, _ = get_run_data(run.info.run_id)

        assert metrics["accuracy_score_eval1_X"] == accuracy_score
        assert metrics["GridSearchCV_score_eval1_X"] == cv_score


def test_gen_metric_call_commands():
    def metric_fn1(a1, b1, *, c2=3, d1=None, d2=True, d3="abc", **kwargs):
        pass

    cmd1 = mlflow.sklearn._AutologgingMetricsManager.gen_metric_call_command(
        None,
        metric_fn1,
        *[np.array([1.0]), pd.DataFrame(data={"c1": [1]})],
        **{"c2": 4, "d1": None, "d2": False, "d3": "def", "randarg1": "a" * 100, "randarg2": "0.1"},
    )

    assert (
        cmd1 == "metric_fn1(a1=<ndarray>, b1=<DataFrame>, c2=4, d1=None, d2=False, d3='def',"
        " randarg1='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...', randarg2='0.1')"
    )

    data1 = np.array([1.0])
    data2 = pd.DataFrame(data={"c1": [1]})

    cmd2 = mlflow.sklearn._AutologgingMetricsManager.gen_metric_call_command(
        None, metric_fn1, *[data1, data2], **{"randarg1": "'xyz\"abc"}
    )
    assert cmd2 == "metric_fn1(a1=data1, b1=data2, randarg1='\\'xyz\"abc')"

    lr_model = sklearn.linear_model.LinearRegression()
    cmd3 = mlflow.sklearn._AutologgingMetricsManager.gen_metric_call_command(
        lr_model, sklearn.linear_model.LinearRegression.score, data1, data2
    )

    assert cmd3 == "LinearRegression.score(X=data1, y=data2)"


def test_patch_for_delegated_method():
    from tests.autologging.test_autologging_utils import get_func_attrs

    original_predict = sklearn.pipeline.Pipeline.predict
    mlflow.sklearn.autolog()

    assert get_func_attrs(sklearn.pipeline.Pipeline.predict) == get_func_attrs(original_predict)

    estimators = [
        ("svc", sklearn.svm.SVC()),
    ]
    model = sklearn.pipeline.Pipeline(estimators)
    X, y = get_iris()

    with mlflow.start_run():
        model.fit(X, y)

    eval1_X = X[0::3]

    with mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.register_prediction_input_dataset"
    ) as mock_register_prediction_input_dataset:
        pred1_y = model.predict(eval1_X)
        # assert `register_prediction_input_dataset` was called and called only once.
        # the `pipeline.predict` call nested `svc.predict`, but sklearn patching function
        # will disable nested call autologging, so the autolog routine is only enabled
        # at `pipeline.predict` level.
        assert mock_register_prediction_input_dataset.call_count <= 1

    mlflow.sklearn.autolog(disable=True)
    pred1_y_original = model.predict(eval1_X)

    np.testing.assert_allclose(pred1_y, pred1_y_original)


@pytest.mark.skipif("Version(sklearn.__version__) <= Version('0.24.2')")
def test_patch_for_available_if_decorated_method():
    from tests.autologging.test_autologging_utils import get_func_attrs

    original_transform = sklearn.pipeline.Pipeline.transform
    mlflow.sklearn.autolog()

    assert get_func_attrs(sklearn.pipeline.Pipeline.transform) == get_func_attrs(original_transform)

    estimators = [
        ("kmeans", sklearn.cluster.KMeans()),
    ]
    model = sklearn.pipeline.Pipeline(estimators)
    X, y = get_iris()

    with mlflow.start_run():
        model.fit(X, y)

    eval1_X = X[0::3]
    transform1_y = model.transform(eval1_X)

    mlflow.sklearn.autolog(disable=True)

    transform1_y_original = model.transform(eval1_X)

    np.testing.assert_allclose(transform1_y, transform1_y_original)


def test_is_metrics_value_loggable():
    is_metric_value_loggable = mlflow.sklearn._AutologgingMetricsManager.is_metric_value_loggable
    assert is_metric_value_loggable(3)
    assert is_metric_value_loggable(3.5)
    assert is_metric_value_loggable(np.float32(3.5))
    assert not is_metric_value_loggable(True)
    assert not is_metric_value_loggable([1, 2])
    assert not is_metric_value_loggable(np.array([1, 2]))


def test_log_post_training_metrics_configuration():
    from sklearn.linear_model import LogisticRegression

    X, y = get_iris()
    model = LogisticRegression()
    metric_name = sklearn.metrics.r2_score.__name__

    # Ensure post-traning metrics autologging can be toggled on / off
    for log_post_training_metrics in [True, False, True]:
        mlflow.sklearn.autolog(log_post_training_metrics=log_post_training_metrics)

        with mlflow.start_run() as run:
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn.metrics.r2_score(y, y_pred)

        metrics = get_run_data(run.info.run_id)[1]
        assert any(k.startswith(metric_name) for k in metrics.keys()) is log_post_training_metrics


class UnpicklableKmeans(sklearn.cluster.KMeans):
    def __init__(self, n_clusters=8):
        super().__init__(n_clusters)
        self.generator = (i for i in range(3))

    # Ignore parameter validation added in scikit-learn > 1.1.0
    def _validate_params(self):
        pass


def test_autolog_print_warning_if_custom_estimator_pickling_raise_error():
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run, mock.patch("mlflow.sklearn._logger.warning") as mock_warning:
        unpicklable_kmeans = UnpicklableKmeans()
        with pytest.raises(TypeError, match=r"(can't|cannot) pickle.+generator"):
            pickle.dumps(unpicklable_kmeans)

        unpicklable_kmeans.fit(*get_iris())
        assert any(
            call_args[0][0].startswith("Pickling custom sklearn model UnpicklableKmeans failed")
            for call_args in mock_warning.call_args_list
        )

    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert len(params) > 0
    assert len(metrics) > 0
    assert len(tags) > 0
    assert artifacts == (["estimator.html"] if _is_estimator_html_repr_supported() else [])


def test_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"

    mlflow.sklearn.autolog(registered_model_name=registered_model_name)
    with mlflow.start_run():
        sklearn.cluster.KMeans().fit(*get_iris())

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


def test_autolog_pos_label_used_for_training_metric():
    mlflow.sklearn.autolog(pos_label=1)

    import sklearn.ensemble

    model = sklearn.ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    with mlflow.start_run() as run:
        model = fit_model(model, X, y, "fit")
        _, training_metrics, _, _ = get_run_data(run.info.run_id)
        with MlflowAutologgingQueueingClient() as autologging_client:
            expected_training_metrics = _log_estimator_content(
                autologging_client=autologging_client,
                estimator=model,
                run_id=run.info.run_id,
                prefix="training_",
                X=X,
                y_true=y,
                sample_weight=None,
                pos_label=1,
            )
    assert training_metrics == expected_training_metrics


def test_autolog_emits_warning_message_when_pos_label_used_for_multilabel():
    mlflow.sklearn.autolog(pos_label=1)

    model = sklearn.svm.SVC()
    X, y = get_iris()

    with mlflow.start_run(), mock.patch("mlflow.sklearn.utils._logger.warning") as mock_warning:
        model.fit(X, y)
        assert mock_warning.call_count == 3  # for precision, recall and f1_score
        mock_warning.assert_any_call(
            "precision_score failed. The metric training_precision_score will not be recorded. "
            "Metric error: Target is multiclass but average='binary'. Please choose another "
            "average setting, one of [None, 'micro', 'macro', 'weighted']."
        )
