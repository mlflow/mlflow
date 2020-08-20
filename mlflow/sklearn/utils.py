from distutils.version import LooseVersion
import inspect
from itertools import islice

_SAMPLE_WEIGHT = "sample_weight"


def _is_old_version():
    import sklearn

    return LooseVersion(sklearn.__version__) < LooseVersion("0.20.3")


def _all_estimators():
    try:
        from sklearn.utils import all_estimators

        return all_estimators()
    except ImportError:
        return _backported_all_estimators()


def _get_Xy(args, kwargs):
    if len(args) >= 2:
        return args[:2]

    if len(args) == 1:
        return args[0], kwargs["y"]

    return kwargs["X"], kwargs["y"]


def _get_sample_weight(args_list, args, kwargs):
    sample_weight_index = args_list.index(_SAMPLE_WEIGHT)

    if len(args) > sample_weight_index:
        return args[sample_weight_index]

    if _SAMPLE_WEIGHT in kwargs:
        return kwargs[_SAMPLE_WEIGHT]

    return None


def _get_arg_names(f):
    # `inspect.getargspec` doesn't return a wrapped function's argspec
    # See: https://hynek.me/articles/decorators/
    return list(inspect.signature(f).parameters.keys())


def _get_args_for_score(fit_func, score_func, args, kwargs):
    fit_args = _get_arg_names(fit_func)
    score_args = _get_arg_names(score_func)

    Xy = _get_Xy(args, kwargs)

    if (_SAMPLE_WEIGHT in score_args) and (_SAMPLE_WEIGHT in fit_args):
        sample_weight = _get_sample_weight(fit_args, args, kwargs)
        return (*Xy, sample_weight)

    return Xy


def _chunk_dict(d, chunk_size):
    # Copied from: https://stackoverflow.com/a/22878842

    it = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}


def _truncate_dict_values(d, max_length):
    return {k: str(v)[:max_length] for k, v in d.items()}


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
