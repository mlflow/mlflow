import collections
from distutils.version import LooseVersion
from itertools import islice
import inspect
import logging
from numbers import Number
import numpy as np
import time

from mlflow.entities import Metric, Param
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import try_mlflow_log
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
    MAX_ENTITIES_PER_BATCH,
    MAX_ENTITY_KEY_LENGTH,
    MAX_PARAM_VAL_LENGTH,
)

_logger = logging.getLogger(__name__)

# The earliest version we're guaranteed to support. Autologging utilities may not work properly
# on scikit-learn older than this version.
_MIN_SKLEARN_VERSION = "0.20.3"

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


def _get_estimator_info_tags(estimator):
    """
    :return: A dictionary of MLflow run tag keys and values
             describing the specified estimator.
    """
    return {
        "estimator_name": estimator.__class__.__name__,
        "estimator_class": (estimator.__class__.__module__ + "." + estimator.__class__.__name__),
    }


def _get_Xy(args, kwargs, X_var_name, y_var_name):
    # corresponds to: model.fit(X, y)
    if len(args) >= 2:
        return args[:2]

    # corresponds to: model.fit(X, <y_var_name>=y)
    if len(args) == 1:
        return args[0], kwargs[y_var_name]

    # corresponds to: model.fit(<X_var_name>=X, <y_var_name>=y)
    return kwargs[X_var_name], kwargs[y_var_name]


def _get_samples_labels_and_predictions(fitted_estimator, fit_args, fit_kwargs, fit_arg_names):
    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    X_var_name, y_var_name = fit_arg_names[:2]
    X, y_true = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)
    y_pred = fitted_estimator.predict(X)

    return X, y_true, y_pred


def _get_sample_weight(arg_names, args, kwargs):
    sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)

    # corresponds to: model.fit(X, y, ..., sample_weight)
    if len(args) > sample_weight_index:
        return args[sample_weight_index]

    # corresponds to: model.fit(X, y, ..., sample_weight=sample_weight)
    if _SAMPLE_WEIGHT in kwargs:
        return kwargs[_SAMPLE_WEIGHT]

    return None


def _get_arg_names(f):
    # `inspect.getargspec` doesn't return a wrapped function's argspec
    # See: https://hynek.me/articles/decorators#mangled-signatures
    return list(inspect.signature(f).parameters.keys())


def _get_args_for_score(score_func, fit_func, fit_args, fit_kwargs):
    """
    Get arguments to pass to score_func in the following steps.

    1. Extract X and y from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in both score_func and fit_func,
       extract it from fit_args or fit_kwargs and return (X, y, sample_weight),
       otherwise return (X, y)

    :param score_func: A score function object.
    :param fit_func: A fit function object.
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.

    :returns: A tuple of either (X, y, sample_weight) or (X, y).
    """
    score_arg_names = _get_arg_names(score_func)
    fit_arg_names = _get_arg_names(fit_func)

    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    # E.g., see: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier.fit # noqa: E501
    X_var_name, y_var_name = fit_arg_names[:2]
    Xy = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)

    if (_SAMPLE_WEIGHT in fit_arg_names) and (_SAMPLE_WEIGHT in score_arg_names):
        sample_weight = _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        return (*Xy, sample_weight)

    return Xy


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


def _get_classifier_metrics(fitted_estimator, fit_args, fit_kwargs):
    """
    Compute and record various common metrics for classifiers

    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    By default, we choose the parameter `labels` to be `None`, `pos_label` to be `1`,
    `average` to be `weighted` to compute the weighted precision score.

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

    :param fitted_estimator: The already fitted classifier
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    fit_arg_names = _get_arg_names(fitted_estimator.fit)
    X, y_true, y_pred = _get_samples_labels_and_predictions(
        fitted_estimator, fit_args, fit_kwargs, fit_arg_names
    )
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if _SAMPLE_WEIGHT in fit_arg_names
        else None
    )

    classifier_metrics = [
        _SklearnMetric(
            name=_TRAINING_PREFIX + "precision_score",
            function=sklearn.metrics.precision_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, average="weighted", sample_weight=sample_weight
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "recall_score",
            function=sklearn.metrics.recall_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, average="weighted", sample_weight=sample_weight
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "f1_score",
            function=sklearn.metrics.f1_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, average="weighted", sample_weight=sample_weight
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "accuracy_score",
            function=sklearn.metrics.accuracy_score,
            arguments=dict(
                y_true=y_true, y_pred=y_pred, normalize=True, sample_weight=sample_weight
            ),
        ),
    ]

    if hasattr(fitted_estimator, "predict_proba"):
        y_pred_proba = fitted_estimator.predict_proba(X)
        classifier_metrics.extend(
            [
                _SklearnMetric(
                    name=_TRAINING_PREFIX + "log_loss",
                    function=sklearn.metrics.log_loss,
                    arguments=dict(y_true=y_true, y_pred=y_pred_proba, sample_weight=sample_weight),
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
                        name=_TRAINING_PREFIX + "roc_auc_score",
                        function=sklearn.metrics.roc_auc_score,
                        arguments=dict(
                            y_true=y_true,
                            y_score=y_pred_proba,
                            average="weighted",
                            sample_weight=sample_weight,
                            multi_class="ovo",
                        ),
                    ),
                ]
            )

    return _get_metrics_value_dict(classifier_metrics)


def _get_classifier_artifacts(fitted_estimator, fit_args, fit_kwargs):
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

    :param fitted_estimator: The already fitted regressor
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: List of artifacts to be logged
    """
    import sklearn

    fit_arg_names = _get_arg_names(fitted_estimator.fit)
    X, y_true, y_pred = _get_samples_labels_and_predictions(
        fitted_estimator, fit_args, fit_kwargs, fit_arg_names
    )
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if _SAMPLE_WEIGHT in fit_arg_names
        else None
    )

    classifier_artifacts = []
    if _is_plotting_supported():
        labels = set(y_true)
        classifier_artifacts.extend(
            [
                _SklearnArtifact(
                    name=_TRAINING_PREFIX + "confusion_matrix",
                    function=sklearn.metrics.plot_confusion_matrix,
                    arguments=dict(
                        estimator=fitted_estimator,
                        X=X,
                        y_true=y_pred,
                        sample_weight=sample_weight,
                        normalize="true",
                        cmap="Blues",
                    ),
                    title="Normalized confusion matrix",
                ),
            ]
        )

        # The plot_roc_curve and plot_precision_recall_curve can only be
        # supported for binary classifier
        if len(labels) == 2:
            classifier_artifacts.extend(
                [
                    _SklearnArtifact(
                        name=_TRAINING_PREFIX + "roc_curve",
                        function=sklearn.metrics.plot_roc_curve,
                        arguments=dict(
                            estimator=fitted_estimator, X=X, y=y_pred, sample_weight=sample_weight,
                        ),
                        title="ROC curve",
                    ),
                    _SklearnArtifact(
                        name=_TRAINING_PREFIX + "precision_recall_curve",
                        function=sklearn.metrics.plot_precision_recall_curve,
                        arguments=dict(
                            estimator=fitted_estimator, X=X, y=y_pred, sample_weight=sample_weight,
                        ),
                        title="Precision recall curve",
                    ),
                ]
            )

    return classifier_artifacts


def _get_regressor_metrics(fitted_estimator, fit_args, fit_kwargs):
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

    :param fitted_estimator: The already fitted regressor
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    fit_arg_names = _get_arg_names(fitted_estimator.fit)
    _, y_true, y_pred = _get_samples_labels_and_predictions(
        fitted_estimator, fit_args, fit_kwargs, fit_arg_names
    )
    sample_weight = (
        _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        if _SAMPLE_WEIGHT in fit_arg_names
        else None
    )

    regressor_metrics = [
        _SklearnMetric(
            name=_TRAINING_PREFIX + "mse",
            function=sklearn.metrics.mean_squared_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "mae",
            function=sklearn.metrics.mean_absolute_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
        _SklearnMetric(
            name=_TRAINING_PREFIX + "r2_score",
            function=sklearn.metrics.r2_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=sample_weight,
                multioutput="uniform_average",
            ),
        ),
    ]

    # To be compatible with older versions of scikit-learn (below 0.22.2), where
    # `sklearn.metrics.mean_squared_error` does not have "squared" parameter to calculate `rmse`,
    # we compute it through np.sqrt(<value of mse>)
    metrics_value_dict = _get_metrics_value_dict(regressor_metrics)
    metrics_value_dict[_TRAINING_PREFIX + "rmse"] = np.sqrt(
        metrics_value_dict[_TRAINING_PREFIX + "mse"]
    )

    return metrics_value_dict


def _log_warning_for_metrics(func_name, func_call, err):
    msg = (
        func_call.__qualname__
        + " failed. The metric "
        + func_name
        + "will not be recorded."
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


def _log_specialized_estimator_content(fitted_estimator, run_id, fit_args, fit_kwargs):
    import sklearn

    mlflow_client = MlflowClient()
    name_metric_dict = {}
    try:
        if sklearn.base.is_classifier(fitted_estimator):
            name_metric_dict = _get_classifier_metrics(fitted_estimator, fit_args, fit_kwargs)

        elif sklearn.base.is_regressor(fitted_estimator):
            name_metric_dict = _get_regressor_metrics(fitted_estimator, fit_args, fit_kwargs)
    except Exception as err:
        msg = (
            "Failed to autolog metrics for "
            + fitted_estimator.__class__.__name__
            + ". Logging error: "
            + str(err)
        )
        _logger.warning(msg)
    else:
        # batch log all metrics
        try_mlflow_log(
            mlflow_client.log_batch,
            run_id,
            metrics=[
                Metric(key=str(key), value=value, timestamp=int(time.time() * 1000), step=0)
                for key, value in name_metric_dict.items()
            ],
        )

    if sklearn.base.is_classifier(fitted_estimator):
        try:
            artifacts = _get_classifier_artifacts(fitted_estimator, fit_args, fit_kwargs)
        except Exception as e:
            msg = (
                "Failed to autolog artifacts for "
                + fitted_estimator.__class__.__name__
                + ". Logging error: "
                + str(e)
            )
            _logger.warning(msg)
            return

        with TempDir() as tmp_dir:
            for artifact in artifacts:
                try:
                    display = artifact.function(**artifact.arguments)
                    display.ax_.set_title(artifact.title)
                    filepath = tmp_dir.path("{}.png".format(artifact.name))
                    display.figure_.savefig(filepath)
                    import matplotlib.pyplot as plt

                    plt.close(display.figure_)
                except Exception as e:
                    _log_warning_for_artifacts(artifact.name, artifact.function, e)

            try_mlflow_log(mlflow_client.log_artifacts, run_id, tmp_dir.path())


def _chunk_dict(d, chunk_size):
    # Copied from: https://stackoverflow.com/a/22878842

    it = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}


def _truncate_dict(d, max_key_length=None, max_value_length=None):
    def _truncate_and_ellipsize(value, max_length):
        return str(value)[: (max_length - 3)] + "..."

    key_is_none = max_key_length is None
    val_is_none = max_value_length is None

    if key_is_none and val_is_none:
        raise ValueError("Must specify at least either `max_key_length` or `max_value_length`")

    truncated = {}
    for k, v in d.items():
        should_truncate_key = (not key_is_none) and (len(str(k)) > max_key_length)
        should_truncate_val = (not val_is_none) and (len(str(v)) > max_value_length)

        new_k = _truncate_and_ellipsize(k, max_key_length) if should_truncate_key else k
        if should_truncate_key:
            # Use the truncated key for warning logs to avoid noisy printing to stdout
            msg = "Truncated the key `{}`".format(new_k)
            _logger.warning(msg)

        new_v = _truncate_and_ellipsize(v, max_value_length) if should_truncate_val else v
        if should_truncate_val:
            # Use the truncated key and value for warning logs to avoid noisy printing to stdout
            msg = "Truncated the value of the key `{}`. Truncated value: `{}`".format(new_k, new_v)
            _logger.warning(msg)

        truncated[new_k] = new_v

    return truncated


def _get_meta_estimators_for_autologging():
    """
    :return: A list of meta estimator class definitions
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
    :return: `True` if the specified scikit-learn estimator is a parameter search estimator,
             such as `GridSearchCV`. `False` otherwise.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    parameter_search_estimators = [
        GridSearchCV,
        RandomizedSearchCV,
    ]

    return any(
        [
            isinstance(estimator, param_search_estimator)
            for param_search_estimator in parameter_search_estimators
        ]
    )


def _log_parameter_search_results_as_artifact(cv_results_df, run_id):
    """
    Records a collection of parameter search results as an MLflow artifact
    for the specified run.

    :param cv_results_df: A Pandas DataFrame containing the results of a parameter search
                          training session, which may be obtained by parsing the `cv_results_`
                          attribute of a trained parameter search estimator such as
                          `GridSearchCV`.
    :param run_id: The ID of the MLflow Run to which the artifact should be recorded.
    """
    with TempDir() as t:
        results_path = t.path("cv_results.csv")
        cv_results_df.to_csv(results_path, index=False)
        try_mlflow_log(MlflowClient().log_artifact, run_id, results_path)


def _create_child_runs_for_parameter_search(cv_estimator, parent_run, child_tags=None):
    """
    Creates a collection of child runs for a parameter search training session.
    Runs are reconstructed from the `cv_results_` attribute of the specified trained
    parameter search estimator - `cv_estimator`, which provides relevant performance
    metrics for each point in the parameter search space. One child run is created
    for each point in the parameter search space. For additional information, see
    `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html`_. # noqa: E501

    :param cv_estimator: The trained parameter search estimator for which to create
                         child runs.
    :param parent_run: A py:class:`mlflow.entities.Run` object referring to the parent
                       parameter search run for which child runs should be created.
    :param child_tags: An optional dictionary of MLflow tag keys and values to log
                       for each child run.
    """
    import pandas as pd

    client = MlflowClient()
    # Use the start time of the parent parameter search run as a rough estimate for the
    # start time of child runs, since we cannot precisely determine when each point
    # in the parameter search space was explored
    child_run_start_time = parent_run.info.start_time
    child_run_end_time = int(time.time() * 1000)

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
    for _, result_row in cv_results_df.iterrows():
        tags_to_log = dict(child_tags) if child_tags else {}
        tags_to_log.update({MLFLOW_PARENT_RUN_ID: parent_run.info.run_id})
        tags_to_log.update(_get_estimator_info_tags(seed_estimator))
        child_run = client.create_run(
            experiment_id=parent_run.info.experiment_id,
            start_time=child_run_start_time,
            tags=tags_to_log,
        )

        from itertools import zip_longest

        params_to_log = dict(base_params)
        params_to_log.update(result_row.get("params", {}))
        param_batches_to_log = _chunk_dict(params_to_log, chunk_size=MAX_PARAMS_TAGS_PER_BATCH)

        # Parameters values are recorded twice in the set of search `cv_results_`:
        # once within a `params` column with dictionary values and once within
        # a separate dataframe column that is created for each parameter. To prevent
        # duplication of parameters, we log the consolidated values from the parameter
        # dictionary column and filter out the other parameter-specific columns with
        # names of the form `param_{param_name}`. Additionally, `cv_results_` produces
        # metrics for each training split, which is fairly verbose; accordingly, we filter
        # out per-split metrics in favor of aggregate metrics (mean, std, etc.)
        excluded_metric_prefixes = ["param", "split"]
        metric_batches_to_log = _chunk_dict(
            {
                key: value
                for key, value in result_row.iteritems()
                if not any([key.startswith(prefix) for prefix in excluded_metric_prefixes])
                and isinstance(value, Number)
            },
            chunk_size=min(
                MAX_ENTITIES_PER_BATCH - MAX_PARAMS_TAGS_PER_BATCH, MAX_METRICS_PER_BATCH
            ),
        )

        for params_batch, metrics_batch in zip_longest(
            param_batches_to_log, metric_batches_to_log, fillvalue={}
        ):
            # Trim any parameter keys / values and metric keys that exceed the limits
            # imposed by corresponding MLflow Tracking APIs (e.g., LogParam, LogMetric)
            truncated_params_batch = _truncate_dict(
                params_batch, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH
            )
            truncated_metrics_batch = _truncate_dict(
                metrics_batch, max_key_length=MAX_ENTITY_KEY_LENGTH
            )
            client.log_batch(
                run_id=child_run.info.run_id,
                params=[
                    Param(str(key), str(value)) for key, value in truncated_params_batch.items()
                ],
                metrics=[
                    Metric(key=str(key), value=value, timestamp=child_run_end_time, step=0)
                    for key, value in truncated_metrics_batch.items()
                ],
            )

        client.set_terminated(run_id=child_run.info.run_id, end_time=child_run_end_time)


def _is_supported_version():
    import sklearn

    return LooseVersion(sklearn.__version__) >= LooseVersion(_MIN_SKLEARN_VERSION)


# Util function to check whether a metric is able to be computed in given sklearn version
def _is_metric_supported(metric_name):
    import sklearn

    # This dict can be extended to store special metrics' specific supported versions
    _metric_supported_version = {"roc_auc_score": "0.22.2"}

    return LooseVersion(sklearn.__version__) >= LooseVersion(_metric_supported_version[metric_name])


# Util function to check whether artifact plotting functions are able to be computed
# in given sklearn version (should >= 0.22.0)
def _is_plotting_supported():
    import sklearn

    return LooseVersion(sklearn.__version__) >= LooseVersion("0.22.0")


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
        and ``class`` is the actuall type of the class.
    """
    # lazy import to avoid circular imports from sklearn.base
    import pkgutil
    import platform
    import sklearn
    from importlib import import_module
    from operator import itemgetter

    # pylint: disable=no-name-in-module, import-error
    from sklearn.utils.testing import ignore_warnings
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin,
        TransformerMixin,
        ClusterMixin,
    )

    IS_PYPY = platform.python_implementation() == "PyPy"

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
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
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
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
                " %s." % repr(type_filter)
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))
