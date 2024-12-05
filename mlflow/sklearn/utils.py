import collections
import inspect
import logging
import pkgutil
import platform
import warnings
from copy import deepcopy
from importlib import import_module
from numbers import Number
from operator import itemgetter

import numpy as np
from packaging.version import Version

from mlflow import MlflowClient
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

# The prefix to note that all calculated metrics and artifacts are solely based on training datasets
_TRAINING_PREFIX = "training_"

_SAMPLE_WEIGHT = "sample_weight"

# _SklearnArtifact represents a artifact (e.g confusion matrix) that will be computed and
# logged during the autologging routine for a particular model type (eg, classifier, regressor).
_SklearnArtifact = collections.namedtuple(
    "_SklearnArtifact", ["name", "function", "arguments", "title"]
)

# _SklearnMetric represents a metric (e.g, precision_score) that will be computed and
# logged during the autologging routine for a particular model type (eg, classifier, regressor).
_SklearnMetric = collections.namedtuple("_SklearnMetric", ["name", "function", "arguments"])


def _gen_xgboost_sklearn_estimators_to_patch():
    import xgboost as xgb

    all_classes = inspect.getmembers(xgb.sklearn, inspect.isclass)
    base_class = xgb.sklearn.XGBModel
    sklearn_estimators = []
    for _, class_object in all_classes:
        if issubclass(class_object, base_class) and class_object != base_class:
            sklearn_estimators.append(class_object)

    return sklearn_estimators


def _gen_lightgbm_sklearn_estimators_to_patch():
    import lightgbm as lgb

    import mlflow.lightgbm

    all_classes = inspect.getmembers(lgb.sklearn, inspect.isclass)
    base_class = lgb.sklearn._LGBMModelBase
    sklearn_estimators = []
    for _, class_object in all_classes:
        package_name = class_object.__module__.split(".")[0]
        if (
            package_name == mlflow.lightgbm.FLAVOR_NAME
            and issubclass(class_object, base_class)
            and class_object != base_class
        ):
            sklearn_estimators.append(class_object)

    return sklearn_estimators


def _get_estimator_info_tags(estimator):
    """
    Returns:
        A dictionary of MLflow run tag keys and values describing the specified estimator.
    """
    return {
        "estimator_name": estimator.__class__.__name__,
        "estimator_class": (estimator.__class__.__module__ + "." + estimator.__class__.__name__),
    }


def _get_X_y_and_sample_weight(fit_func, fit_args, fit_kwargs):
    """
    Get a tuple of (X, y, sample_weight) in the following steps.

    1. Extract X and y from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in fit_func,
       extract it from fit_args or fit_kwargs and return (X, y, sample_weight),
       otherwise return (X, y)

    Args:
        fit_func: A fit function object.
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        A tuple of either (X, y, sample_weight), where `y` and `sample_weight` may be
        `None` if the specified `fit_args` and `fit_kwargs` do not specify labels or
        a sample weighting. Copies of `X` and `y` are made in order to avoid mutation
        of the dataset during training.
    """

    def _get_Xy(args, kwargs, X_var_name, y_var_name):
        # corresponds to: model.fit(X, y)
        if len(args) >= 2:
            return args[:2]

        # corresponds to: model.fit(X, <y_var_name>=y)
        if len(args) == 1:
            return args[0], kwargs.get(y_var_name)

        # corresponds to: model.fit(<X_var_name>=X, <y_var_name>=y)
        return kwargs[X_var_name], kwargs.get(y_var_name)

    def _get_sample_weight(arg_names, args, kwargs):
        sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)

        # corresponds to: model.fit(X, y, ..., sample_weight)
        if len(args) > sample_weight_index:
            return args[sample_weight_index]

        # corresponds to: model.fit(X, y, ..., sample_weight=sample_weight)
        if _SAMPLE_WEIGHT in kwargs:
            return kwargs[_SAMPLE_WEIGHT]

        return None

    fit_arg_names = _get_arg_names(fit_func)
    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    # E.g., see: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier.fit
    X_var_name, y_var_name = fit_arg_names[:2]
    X, y = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)
    if X is not None:
        X = deepcopy(X)
    if y is not None:
        y = deepcopy(y)
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if (_SAMPLE_WEIGHT in fit_arg_names)
        else None
    )

    return (X, y, sample_weight)


def _get_metrics_value_dict(metrics_list):
    metric_value_dict = {}
    for metric in metrics_list:
        try:
            metric_value = metric.function(**metric.arguments)
        except Exception as e:
            _log_warning_for_metrics(metric.name, metric.function, e)
        else:
            metric_value_dict[metric.name] = metric_value
    return metric_value_dict


def _get_classifier_metrics(fitted_estimator, prefix, X, y_true, sample_weight, pos_label):  # noqa: D417
    """
    Compute and record various common metrics for classifiers

    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    By default, when `pos_label` is not specified (passed in as `None`), we set `average`
    to `weighted` to compute the weighted score of these metrics.
    When the `pos_label` is specified (not `None`), we set `average` to `binary`.

    For (4) accuracy score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    we choose the parameter `normalize` to be `True` to output the percentage of accuracy,
    as opposed to `False` that outputs the absolute correct number of sample prediction

    We log additional metrics if certain classifier has method `predict_proba`
    (5) log loss:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    (6) roc_auc_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    By default, for roc_auc_score, we pick `average` to be `weighted`, `multi_class` to be `ovo`,
    to make the output more insensitive to dataset imbalance.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, ...... sample_weight), otherwise as (y_true, y_pred, ......)
    3. return a dictionary of metric(name, value)

    Args:
        fitted_estimator: The already fitted classifier
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        dictionary of (function name, computed value)
    """
    import sklearn

    average = "weighted" if pos_label is None else "binary"
    y_pred = fitted_estimator.predict(X)

    classifier_metrics = [
        _SklearnMetric(
            name=prefix + "precision_score",
            function=sklearn.metrics.precision_score,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "pos_label": pos_label,
                "average": average,
                "sample_weight": sample_weight,
            },
        ),
        _SklearnMetric(
            name=prefix + "recall_score",
            function=sklearn.metrics.recall_score,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "pos_label": pos_label,
                "average": average,
                "sample_weight": sample_weight,
            },
        ),
        _SklearnMetric(
            name=prefix + "f1_score",
            function=sklearn.metrics.f1_score,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "pos_label": pos_label,
                "average": average,
                "sample_weight": sample_weight,
            },
        ),
        _SklearnMetric(
            name=prefix + "accuracy_score",
            function=sklearn.metrics.accuracy_score,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "normalize": True,
                "sample_weight": sample_weight,
            },
        ),
    ]

    if hasattr(fitted_estimator, "predict_proba"):
        y_pred_proba = fitted_estimator.predict_proba(X)
        classifier_metrics.extend(
            [
                _SklearnMetric(
                    name=prefix + "log_loss",
                    function=sklearn.metrics.log_loss,
                    arguments={
                        "y_true": y_true,
                        "y_pred": y_pred_proba,
                        "sample_weight": sample_weight,
                    },
                ),
            ]
        )

        if _is_metric_supported("roc_auc_score"):
            # For binary case, the parameter `y_score` expect scores must be
            # the scores of the class with the greater label.
            if len(y_pred_proba[0]) == 2:
                y_pred_proba = y_pred_proba[:, 1]

            classifier_metrics.extend(
                [
                    _SklearnMetric(
                        name=prefix + "roc_auc",
                        function=sklearn.metrics.roc_auc_score,
                        arguments={
                            "y_true": y_true,
                            "y_score": y_pred_proba,
                            "average": "weighted",
                            "sample_weight": sample_weight,
                            "multi_class": "ovo",
                        },
                    ),
                ]
            )

    return _get_metrics_value_dict(classifier_metrics)


def _get_class_labels_from_estimator(estimator):
    """
    Extracts class labels from `estimator` if `estimator.classes` is available.
    """
    return estimator.classes_ if hasattr(estimator, "classes_") else None


def _get_classifier_artifacts(fitted_estimator, prefix, X, y_true, sample_weight):  # noqa: D417
    """
    Draw and record various common artifacts for classifier

    For all classifiers, we always log:
    (1) confusion matrix:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

    For only binary classifiers, we will log:
    (2) precision recall curve:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_precision_recall_curve.html
    (3) roc curve:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and split into train & test datasets.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a list of artifacts path to be logged

    Args:
        fitted_estimator: The already fitted regressor
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        List of artifacts to be logged
    """
    import sklearn

    if not _is_plotting_supported():
        return []

    is_plot_function_deprecated = Version(sklearn.__version__) >= Version("1.0")

    def plot_confusion_matrix(*args, **kwargs):
        import matplotlib
        import matplotlib.pyplot as plt

        class_labels = _get_class_labels_from_estimator(fitted_estimator)
        if class_labels is None:
            class_labels = set(y_true)

        with matplotlib.rc_context(
            {
                "font.size": min(8.0, 50.0 / len(class_labels)),
                "axes.labelsize": 8.0,
                "figure.dpi": 175,
            }
        ):
            _, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
            return (
                sklearn.metrics.ConfusionMatrixDisplay.from_estimator(*args, **kwargs, ax=ax)
                if is_plot_function_deprecated
                else sklearn.metrics.plot_confusion_matrix(*args, **kwargs, ax=ax)
            )

    y_true_arg_name = "y" if is_plot_function_deprecated else "y_true"
    classifier_artifacts = [
        _SklearnArtifact(
            name=prefix + "confusion_matrix",
            function=plot_confusion_matrix,
            arguments=dict(
                estimator=fitted_estimator,
                X=X,
                sample_weight=sample_weight,
                normalize="true",
                cmap="Blues",
                **{y_true_arg_name: y_true},
            ),
            title="Normalized confusion matrix",
        ),
    ]

    # The plot_roc_curve and plot_precision_recall_curve can only be
    # supported for binary classifier
    if len(set(y_true)) == 2:
        classifier_artifacts.extend(
            [
                _SklearnArtifact(
                    name=prefix + "roc_curve",
                    function=sklearn.metrics.RocCurveDisplay.from_estimator
                    if is_plot_function_deprecated
                    else sklearn.metrics.plot_roc_curve,
                    arguments={
                        "estimator": fitted_estimator,
                        "X": X,
                        "y": y_true,
                        "sample_weight": sample_weight,
                    },
                    title="ROC curve",
                ),
                _SklearnArtifact(
                    name=prefix + "precision_recall_curve",
                    function=sklearn.metrics.PrecisionRecallDisplay.from_estimator
                    if is_plot_function_deprecated
                    else sklearn.metrics.plot_precision_recall_curve,
                    arguments={
                        "estimator": fitted_estimator,
                        "X": X,
                        "y": y_true,
                        "sample_weight": sample_weight,
                    },
                    title="Precision recall curve",
                ),
            ]
        )

    return classifier_artifacts


def _get_regressor_metrics(fitted_estimator, prefix, X, y_true, sample_weight):  # noqa: D417
    """
    Compute and record various common metrics for regressors

    For (1) (root) mean squared error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    (2) mean absolute error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    (3) r2 score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    By default, we choose the parameter `multioutput` to be `uniform_average`
    to average outputs with uniform weight.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs, and compute y_pred.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default
    has sample_weight), extract it from fit_args or fit_kwargs as
    (y_true, y_pred, sample_weight, multioutput), otherwise as (y_true, y_pred, multioutput)
    3. return a dictionary of metric(name, value)

    Args:
        fitted_estimator: The already fitted regressor
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        dictionary of (function name, computed value)
    """
    import sklearn

    y_pred = fitted_estimator.predict(X)

    regressor_metrics = [
        _SklearnMetric(
            name=prefix + "mean_squared_error",
            function=sklearn.metrics.mean_squared_error,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
                "multioutput": "uniform_average",
            },
        ),
        _SklearnMetric(
            name=prefix + "mean_absolute_error",
            function=sklearn.metrics.mean_absolute_error,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
                "multioutput": "uniform_average",
            },
        ),
        _SklearnMetric(
            name=prefix + "r2_score",
            function=sklearn.metrics.r2_score,
            arguments={
                "y_true": y_true,
                "y_pred": y_pred,
                "sample_weight": sample_weight,
                "multioutput": "uniform_average",
            },
        ),
    ]

    # To be compatible with older versions of scikit-learn (below 0.22.2), where
    # `sklearn.metrics.mean_squared_error` does not have "squared" parameter to calculate `rmse`,
    # we compute it through np.sqrt(<value of mse>)
    metrics_value_dict = _get_metrics_value_dict(regressor_metrics)
    metrics_value_dict[prefix + "root_mean_squared_error"] = np.sqrt(
        metrics_value_dict[prefix + "mean_squared_error"]
    )

    return metrics_value_dict


def _log_warning_for_metrics(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The metric "
        + func_name
        + " will not be recorded."
        + " Metric error: "
        + str(err)
    )
    _logger.warning(msg)


def _log_warning_for_artifacts(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The artifact "
        + func_name
        + " will not be recorded."
        + " Artifact error: "
        + str(err)
    )
    _logger.warning(msg)


def _log_specialized_estimator_content(
    autologging_client, fitted_estimator, run_id, prefix, X, y_true, sample_weight, pos_label
):
    import sklearn

    metrics = {}

    if y_true is not None:
        try:
            if sklearn.base.is_classifier(fitted_estimator):
                metrics = _get_classifier_metrics(
                    fitted_estimator, prefix, X, y_true, sample_weight, pos_label
                )
            elif sklearn.base.is_regressor(fitted_estimator):
                metrics = _get_regressor_metrics(fitted_estimator, prefix, X, y_true, sample_weight)
        except Exception as err:
            msg = (
                "Failed to autolog metrics for "
                + fitted_estimator.__class__.__name__
                + ". Logging error: "
                + str(err)
            )
            _logger.warning(msg)
        else:
            autologging_client.log_metrics(run_id=run_id, metrics=metrics)

    if sklearn.base.is_classifier(fitted_estimator):
        try:
            artifacts = _get_classifier_artifacts(
                fitted_estimator, prefix, X, y_true, sample_weight
            )
        except Exception as e:
            msg = (
                "Failed to autolog artifacts for "
                + fitted_estimator.__class__.__name__
                + ". Logging error: "
                + str(e)
            )
            _logger.warning(msg)
            return metrics

        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as ie:
            _logger.warning(
                f"Failed to import matplotlib (error: {ie!r}). Skipping artifact logging."
            )
            return metrics

        _matplotlib_config = {"savefig.dpi": 175, "figure.autolayout": True, "font.size": 8}
        with TempDir() as tmp_dir:
            for artifact in artifacts:
                try:
                    with matplotlib.rc_context(_matplotlib_config):
                        display = artifact.function(**artifact.arguments)
                        display.ax_.set_title(artifact.title)
                        artifact_path = f"{artifact.name}.png"
                        filepath = tmp_dir.path(artifact_path)
                        display.figure_.savefig(fname=filepath, format="png")
                        plt.close(display.figure_)
                except Exception as e:
                    _log_warning_for_artifacts(artifact.name, artifact.function, e)

            MlflowClient().log_artifacts(run_id, tmp_dir.path())

    return metrics


def _is_estimator_html_repr_supported():
    import sklearn

    # Only scikit-learn >= 0.23 supports `estimator_html_repr`
    return Version(sklearn.__version__) >= Version("0.23.0")


def _log_estimator_html(run_id, estimator):
    if not _is_estimator_html_repr_supported():
        return

    from sklearn.utils import estimator_html_repr

    # Specifies charset so triangle toggle buttons are not garbled
    estimator_html_string = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8"/>
  </head>
  <body>
    {estimator_html_repr(estimator)}
  </body>
</html>
    """
    MlflowClient().log_text(run_id, estimator_html_string, artifact_file="estimator.html")


def _log_estimator_content(
    autologging_client,
    estimator,
    run_id,
    prefix,
    X,
    y_true=None,
    sample_weight=None,
    pos_label=None,
):
    """
    Logs content for the given estimator, which includes metrics and artifacts that might be
    tailored to the estimator's type (e.g., regression vs classification). Training labels
    are required for metric computation; metrics will be omitted if labels are not available.

    Args:
        autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
            efficiently logging run data to MLflow Tracking.
        estimator: The estimator used to compute metrics and artifacts.
        run_id: The run under which the content is logged.
        prefix: A prefix used to name the logged content. Typically it's 'training_' for
            training-time content and user-controlled for evaluation-time content.
        X: The data samples.
        y_true: Labels.
        sample_weight: Per-sample weights used in the computation of metrics and artifacts.
        pos_label: The positive label used to compute binary classification metrics such as
            precision, recall, f1, etc. This parameter is only used for classification metrics.
            If set to `None`, the function will calculate metrics for each label and find their
            average weighted by support (number of true instances for each label).

    Returns:
        A dict of the computed metrics.
    """
    metrics = _log_specialized_estimator_content(
        autologging_client=autologging_client,
        fitted_estimator=estimator,
        run_id=run_id,
        prefix=prefix,
        X=X,
        y_true=y_true,
        sample_weight=sample_weight,
        pos_label=pos_label,
    )

    if hasattr(estimator, "score") and y_true is not None:
        try:
            # Use the sample weight only if it is present in the score args
            score_arg_names = _get_arg_names(estimator.score)
            score_args = (
                (X, y_true, sample_weight) if _SAMPLE_WEIGHT in score_arg_names else (X, y_true)
            )
            score = estimator.score(*score_args)
        except Exception as e:
            msg = (
                estimator.score.__qualname__
                + " failed. The 'training_score' metric will not be recorded. Scoring error: "
                + str(e)
            )
            _logger.warning(msg)
        else:
            score_key = prefix + "score"
            autologging_client.log_metrics(run_id=run_id, metrics={score_key: score})
            metrics[score_key] = score
    _log_estimator_html(run_id, estimator)
    return metrics


def _get_meta_estimators_for_autologging():
    """
    Returns:
        A list of meta estimator class definitions
        (e.g., `sklearn.model_selection.GridSearchCV`) that should be included
        when patching training functions for autologging
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    return [
        GridSearchCV,
        RandomizedSearchCV,
        Pipeline,
    ]


def _is_parameter_search_estimator(estimator):
    """
    Returns:
        `True` if the specified scikit-learn estimator is a parameter search estimator,
        such as `GridSearchCV`. `False` otherwise.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    parameter_search_estimators = [
        GridSearchCV,
        RandomizedSearchCV,
    ]

    return any(
        isinstance(estimator, param_search_estimator)
        for param_search_estimator in parameter_search_estimators
    )


def _log_parameter_search_results_as_artifact(cv_results_df, run_id):
    """
    Records a collection of parameter search results as an MLflow artifact
    for the specified run.

    Args:
        cv_results_df: A Pandas DataFrame containing the results of a parameter search
            training session, which may be obtained by parsing the `cv_results_`
            attribute of a trained parameter search estimator such as
            `GridSearchCV`.
        run_id: The ID of the MLflow Run to which the artifact should be recorded.
    """
    with TempDir() as t:
        results_path = t.path("cv_results.csv")
        cv_results_df.to_csv(results_path, index=False)
        MlflowClient().log_artifact(run_id, results_path)


# Log how many child runs will be created vs omitted based on `max_tuning_runs`.
def _log_child_runs_info(max_tuning_runs, total_runs):
    rest = total_runs - max_tuning_runs

    # Set logging statement for runs to be logged.
    if max_tuning_runs == 0:
        logging_phrase = "no runs"
    elif max_tuning_runs == 1:
        logging_phrase = "the best run"
    else:
        logging_phrase = f"the {max_tuning_runs} best runs"

    # Set logging statement for runs to be omitted.
    if rest <= 0:
        omitting_phrase = "no runs"
    elif rest == 1:
        omitting_phrase = "one run"
    else:
        omitting_phrase = f"{rest} runs"

    _logger.info("Logging %s, %s will be omitted.", logging_phrase, omitting_phrase)


def _create_child_runs_for_parameter_search(  # noqa: D417
    autologging_client, cv_estimator, parent_run, max_tuning_runs, child_tags=None
):
    """
    Creates a collection of child runs for a parameter search training session.
    Runs are reconstructed from the `cv_results_` attribute of the specified trained
    parameter search estimator - `cv_estimator`, which provides relevant performance
    metrics for each point in the parameter search space. One child run is created
    for each point in the parameter search space. For additional information, see
    `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html`_.

    Args:
        autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
            efficiently logging run data to MLflow Tracking.
        cv_estimator: The trained parameter search estimator for which to create
            child runs.
        parent_run: A py:class:`mlflow.entities.Run` object referring to the parent
            parameter search run for which child runs should be created.
        child_tags: An optional dictionary of MLflow tag keys and values to log
            for each child run.
    """
    import pandas as pd

    def first_custom_rank_column(df):
        column_names = df.columns.values
        for col_name in column_names:
            if "rank_test_" in col_name:
                return col_name

    # Use the start time of the parent parameter search run as a rough estimate for the
    # start time of child runs, since we cannot precisely determine when each point
    # in the parameter search space was explored
    child_run_start_time = parent_run.info.start_time
    child_run_end_time = get_current_time_millis()

    seed_estimator = cv_estimator.estimator
    # In the unlikely case that a seed of a parameter search estimator is,
    # itself, a parameter search estimator, we should avoid logging the untuned
    # parameters of the seeds's seed estimator
    should_log_params_deeply = not _is_parameter_search_estimator(seed_estimator)
    # Each row of `cv_results_` only provides parameters that vary across
    # the user-specified parameter grid. In order to log the complete set
    # of parameters for each child run, we fetch the parameters defined by
    # the seed estimator and update them with parameter subset specified
    # in the result row
    base_params = seed_estimator.get_params(deep=should_log_params_deeply)
    cv_results_df = pd.DataFrame.from_dict(cv_estimator.cv_results_)

    if max_tuning_runs is None:
        cv_results_best_n_df = cv_results_df
    else:
        rank_column_name = "rank_test_score"
        if rank_column_name not in cv_results_df.columns.values:
            rank_column_name = first_custom_rank_column(cv_results_df)
            warnings.warn(
                f"Top {max_tuning_runs} child runs will be created based on ordering in "
                f"{rank_column_name} column.  You can choose not to limit the number of "
                "child runs created by setting `max_tuning_runs=None`."
            )
        cv_results_best_n_df = cv_results_df.nsmallest(max_tuning_runs, rank_column_name)
        # Log how many child runs will be created vs omitted.
        _log_child_runs_info(max_tuning_runs, len(cv_results_df))

    for _, result_row in cv_results_best_n_df.iterrows():
        tags_to_log = dict(child_tags) if child_tags else {}
        tags_to_log.update({MLFLOW_PARENT_RUN_ID: parent_run.info.run_id})
        tags_to_log.update(_get_estimator_info_tags(seed_estimator))
        pending_child_run_id = autologging_client.create_run(
            experiment_id=parent_run.info.experiment_id,
            start_time=child_run_start_time,
            tags=tags_to_log,
        )

        params_to_log = dict(base_params)
        params_to_log.update(result_row.get("params", {}))
        autologging_client.log_params(run_id=pending_child_run_id, params=params_to_log)

        # Parameters values are recorded twice in the set of search `cv_results_`:
        # once within a `params` column with dictionary values and once within
        # a separate dataframe column that is created for each parameter. To prevent
        # duplication of parameters, we log the consolidated values from the parameter
        # dictionary column and filter out the other parameter-specific columns with
        # names of the form `param_{param_name}`. Additionally, `cv_results_` produces
        # metrics for each training split, which is fairly verbose; accordingly, we filter
        # out per-split metrics in favor of aggregate metrics (mean, std, etc.)
        excluded_metric_prefixes = ["param", "split"]
        metrics_to_log = {
            key: value
            for key, value in result_row.items()
            if not any(key.startswith(prefix) for prefix in excluded_metric_prefixes)
            and isinstance(value, Number)
        }
        autologging_client.log_metrics(
            run_id=pending_child_run_id,
            metrics=metrics_to_log,
        )

        autologging_client.set_terminated(run_id=pending_child_run_id, end_time=child_run_end_time)


# Util function to check whether a metric is able to be computed in given sklearn version
def _is_metric_supported(metric_name):
    import sklearn

    # This dict can be extended to store special metrics' specific supported versions
    _metric_supported_version = {"roc_auc_score": "0.22.2"}

    return Version(sklearn.__version__) >= Version(_metric_supported_version[metric_name])


# Util function to check whether artifact plotting functions are able to be computed
# in given sklearn version (should >= 0.22.0)
def _is_plotting_supported():
    import sklearn

    return Version(sklearn.__version__) >= Version("0.22.0")


def _all_estimators():
    try:
        from sklearn.utils import all_estimators

        return all_estimators()
    except ImportError:
        return _backported_all_estimators()


def _backported_all_estimators(type_filter=None):
    """
    Backported from scikit-learn 0.23.2:
    https://github.com/scikit-learn/scikit-learn/blob/0.23.2/sklearn/utils/__init__.py#L1146

    Use this backported `all_estimators` in old versions of sklearn because:
    1. An inferior version of `all_estimators` that old versions of sklearn use for testing,
       might function differently from a newer version.
    2. This backported `all_estimators` works on old versions of sklearn that don’t even define
       the testing utility variant of `all_estimators`.

    ========== original docstring ==========
    Get a list of all estimators from sklearn.
    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators such as GridSearchCV are also not included.
    Parameters
    ----------
    type_filter : string, list of string,  or None, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    # lazy import to avoid circular imports from sklearn.base
    import sklearn
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        ClusterMixin,
        RegressorMixin,
        TransformerMixin,
    )
    from sklearn.utils._testing import ignore_warnings

    IS_PYPY = platform.python_implementation() == "PyPy"

    def is_abstract(c):
        if not hasattr(c, "__abstractmethods__"):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {"tests", "externals", "setup", "conftest"}
    root = sklearn.__path__[0]  # sklearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, modname, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [(name, est_cls) for name, est_cls in classes if not name.startswith("_")]

            # TODO: Remove when FeatureHasher is implemented in PYPY
            # Skips FeatureHasher for PYPY
            if IS_PYPY and "feature_extraction" in modname:
                classes = [(name, est_cls) for name, est_cls in classes if name == "FeatureHasher"]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c for c in all_classes if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        # copy the object if type_filter is a list
        type_filter = list(type_filter) if isinstance(type_filter, list) else [type_filter]
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend([est for est in estimators if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                "None, got"
                f" {type_filter!r}"
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))
