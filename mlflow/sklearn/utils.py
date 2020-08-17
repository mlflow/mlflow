def _all_estimators(type_filter=None):
    """
    Taken from scikit-learn 0.23.2:
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/utils/__init__.py#L1146

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
    import inspect
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
