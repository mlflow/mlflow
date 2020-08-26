import inspect
import logging
import mlflow
import time

from distutils.version import LooseVersion
from itertools import islice

from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import try_mlflow_log

_logger = logging.getLogger(__name__)

# The earliest version we're guaranteed to support. Autologging utilities may not work properly
# on scikit-learn older than this version.
_MIN_SKLEARN_VERSION = "0.20.3"

_SAMPLE_WEIGHT = "sample_weight"


def _get_Xy(args, kwargs, X_var_name, y_var_name):
    # corresponds to: model.fit(X, y)
    if len(args) >= 2:
        return args[:2]

    # corresponds to: model.fit(X, <y_var_name>=y)
    if len(args) == 1:
        return args[0], kwargs[y_var_name]

    # corresponds to: model.fit(<X_var_name>=X, <y_var_name>=y)
    return kwargs[X_var_name], kwargs[y_var_name]


def _get_fit_Xy(trained_estimator, fit_args, fit_kwargs, fit_arg_names):
    # In most cases, X_var_name and y_var_name become "X" and "y", respectively.
    # However, certain sklearn models use different variable names for X and y.
    X_var_name, y_var_name = fit_arg_names[:2]
    X, y_true = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)
    y_pred = trained_estimator.predict(X)

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
    # See: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso.score # noqa: E501
    X_var_name, y_var_name = fit_arg_names[:2]
    Xy = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)

    if (_SAMPLE_WEIGHT in fit_arg_names) and (_SAMPLE_WEIGHT in score_arg_names):
        sample_weight = _get_sample_weight(fit_arg_names, fit_args, fit_kwargs)
        return (*Xy, sample_weight)

    return Xy


def _get_classifier_metrics(trained_estimator, fit_args, fit_kwargs):
    """
    Compute and log various common metrics for classifiers

    For (1) precision score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    (2) recall score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    (3) f1_score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    By default, we choose the parameter `labels` to be `None`, `pos_label` to be `1`, `average` to be `weighted` to
    compute the weighted precision score.

    For accuracy score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    we choose the parameter `normalize` to be `True` to output the percentage of accuracy
    as opposed to `False` that outputs the absolute correct number of sample prediction

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default has sample_weight),
       extract it from fit_args or fit_kwargs as (y_true, y_pred, ...... sample_weight),
       otherwise as (y_true, y_pred, ......)
    3. return a dictionary of metric(name, value)

    :param trained_estimator: The already fitted classifier
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    fit_arg_names = _get_arg_names(trained_estimator.fit)
    X, y_true, y_pred = _get_fit_Xy(trained_estimator, fit_args, fit_kwargs, fit_arg_names)

    # Maintain a metrics dictionary to store metrics info
    # name_args_func_metrics_dict stores pairs of (function name, (function arguments, function call))
    metrics_dict_classifier = {'precision_score':
                               ((y_true, y_pred, None, 1, 'weighted'), sklearn.metrics.precision_score),
                               'recall_score': ((y_true, y_pred, None, 1, 'weighted'), sklearn.metrics.recall_score),
                               'f1_score': ((y_true, y_pred, None, 1, 'weighted'), sklearn.metrics.f1_score),
                               'accuracy_score': ((y_true, y_pred, True), sklearn.metrics.accuracy_score)}

    name_score_dict = {}
    for func_name, func_args_call in metrics_dict_classifier.items():
        try:
            func_args = func_args_call[0]
            func_call = func_args_call[1]
            sample_weight = _get_sample_weight(fit_arg_names, fit_args, fit_kwargs) \
                if _SAMPLE_WEIGHT in fit_arg_names else None
            func_args = (*func_args, sample_weight)

            func_score = func_call(*func_args)
        except Exception as e:  # pylint: disable=broad-except
            _log_warning(func_name, func_call, e)
        else:
            name_score_dict[func_name] = func_score

    return name_score_dict


def _get_regressor_metrics(trained_estimator, fit_args, fit_kwargs):
    """
    Compute and log various common metrics for regressors

    For (1) (root) mean squared error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    (2) mean absolute error:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
    (3) r2 score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    By default, we choose the parameter `multioutput` to be `uniform_average` to average outputs with uniform weight.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in fit_func (accuracy_score by default has sample_weight),
       extract it from fit_args or fit_kwargs as (y_true, y_pred, sample_weight, multioutput),
       otherwise as (y_true, y_pred, multioutput)
    3. return a dictionary of metric(name, value)

    :param trained_estimator: The already fitted regressor
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    fit_arg_names = _get_arg_names(trained_estimator.fit)
    X, y_true, y_pred = _get_fit_Xy(trained_estimator, fit_args, fit_kwargs, fit_arg_names)

    # Maintain a metrics dictionary to store metrics info
    # name_args_func_metrics_dict stores pairs of (function name, (function arguments, function call))
    metrics_dict_regressor = {'mse': ((y_true, y_pred), sklearn.metrics.mean_squared_error),
                              'rmse': ((y_true, y_pred), sklearn.metrics.mean_squared_error),
                              'mae': ((y_true, y_pred), sklearn.metrics.mean_absolute_error),
                              'r2_score': ((y_true, y_pred), sklearn.metrics.r2_score)}

    name_score_dict = {}
    for func_name, func_args_call in metrics_dict_regressor.items():
        try:
            func_args = func_args_call[0]
            func_call = func_args_call[1]
            sample_weight = _get_sample_weight(fit_arg_names, fit_args, fit_kwargs) \
                if _SAMPLE_WEIGHT in fit_arg_names else None
            # Always add the multioutput default value 'uniform_average'
            func_args = (*func_args, sample_weight, 'uniform_average')

            # For a special case of rmse, the last boolean for parameter 'squared' is needed and should be False
            func_args = (*func_args, False) if (func_name == 'rmse') else func_args
            func_score = func_call(*func_args)
        except Exception as e:  # pylint: disable=broad-except
            _log_warning(func_name, func_call, e)
        else:
            name_score_dict[func_name] = func_score

    return name_score_dict


def _get_clusterer_metrics(trained_estimator, fit_args, fit_kwargs):
    """
    Compute and log various common metrics for clusterers

    For (1) completeness score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
    (2) homogeneity score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
    (3) v-measure score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
    By default, we choose the parameter 'beta' for v-measure score to be 1.0.

    Steps:
    1. Extract X and y_true from fit_args and fit_kwargs.
    2. return a dictionary of metric(name, value)

    :param trained_estimator: The already fitted clusterer
    :param fit_args: Positional arguments given to fit_func.
    :param fit_kwargs: Keyword arguments given to fit_func.
    :return: dictionary of (function name, computed value)
    """
    import sklearn

    print("In Clusterer!")
    fit_arg_names = _get_arg_names(trained_estimator.fit)
    X, y_true, y_pred = _get_fit_Xy(trained_estimator, fit_args, fit_kwargs, fit_arg_names)

    # Maintain a metrics dictionary to store metrics info
    # name_args_func_metrics_dict stores pairs of (function name, (function arguments, function call))
    metrics_dict_clusterer = {'completeness_score': ((y_true, y_pred), sklearn.metrics.completeness_score),
                              'homogeneity_score': ((y_true, y_pred), sklearn.metrics.homogeneity_score),
                              'v_measure_score': ((y_true, y_pred, 1.0), sklearn.metrics.v_measure_score)}

    name_score_dict = {}
    for func_name, func_args_call in metrics_dict_clusterer.items():
        try:
            func_args = func_args_call[0]
            func_call = func_args_call[1]
            func_score = func_call(*func_args)
            print('name {0}, value {1}'.format(func_name, func_score))
        except Exception as e:  # pylint: disable=broad-except
            _log_warning(func_name, func_call, e)
        else:
            name_score_dict[func_name] = func_score

    return name_score_dict


def _log_warning(func_name, func_call, err):
    msg = (
            func_call.__qualname__
            + " failed. The " + func_name + " metric will not be recorded. Error: "
            + str(err)
    )
    print('Exception name {0}'.format(func_name))
    _logger.warning(msg)


def _log_specialized_estimator_content(trained_estimator, run_id, fit_args, fit_kwargs):
    import sklearn

    name_score_dict = {}
    if sklearn.base.is_classifier(trained_estimator):
        name_score_dict = _get_classifier_metrics(trained_estimator, fit_args, fit_kwargs)
    elif sklearn.base.is_regressor(trained_estimator):
        name_score_dict = _get_regressor_metrics(trained_estimator, fit_args, fit_kwargs)
    elif hasattr(trained_estimator, "_estimator_type") and trained_estimator._estimator_type == "clusterer":
        name_score_dict = _get_clusterer_metrics(trained_estimator, fit_args, fit_kwargs)

    # batch log all metrics
    try_mlflow_log(MlflowClient().log_batch, run_id,
                   metrics=[Metric(key=str(key), value=value, timestamp=time.time() * 1000, step=0)
                            for key, value in name_score_dict.items()])


def _chunk_dict(d, chunk_size):
    # Copied from: https://stackoverflow.com/a/22878842

    it = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}


def _truncate_dict(d, max_key_length=None, max_value_length=None):
    key_is_none = max_key_length is None
    val_is_none = max_value_length is None

    if key_is_none and val_is_none:
        raise ValueError("Must specify at least either `max_key_length` or `max_value_length`")

    truncated = {}
    for k, v in d.items():
        should_truncate_key = (not key_is_none) and (len(str(k)) > max_key_length)
        should_truncate_val = (not val_is_none) and (len(str(v)) > max_value_length)

        new_k = str(k)[:max_key_length] if should_truncate_key else k
        if should_truncate_key:
            msg = "Truncated the key `{}`".format(k)
            _logger.warning(msg)

        new_v = str(v)[:max_value_length] if should_truncate_val else v
        if should_truncate_val:
            msg = "Truncated the value `{}` (in the key `{}`)".format(v, k)
            _logger.warning(msg)

        truncated[new_k] = new_v

    return truncated


def _is_supported_version():
    import sklearn

    return LooseVersion(sklearn.__version__) >= LooseVersion(_MIN_SKLEARN_VERSION)


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
    from sklearn.utils.testing import ignore_warnings  # pylint: disable=no-name-in-module
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
