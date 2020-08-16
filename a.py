# import mlflow.sklearn

# mlflow.sklearn.autolog()

# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline

# X, y = make_classification(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
# pipe.fit(X_train, y_train)


# print(StandardScaler.fit.__name__)
# print(StandardScaler.fit.__doc__)

import sklearn
import mlflow.sklearn
from copy import deepcopy
import inspect


def get_func_attrs(f):
    attrs = {}
    for attr_name in ["__doc__", "__name__"]:
        if hasattr(f, attr_name):
            attrs[attr_name] = getattr(f, attr_name)

    attrs["__signature__"] = inspect.signature(f)
    return attrs


def get_fit_attrs(cls):
    attrs = {}
    for method_name in ["fit", "fit_transform", "fit_transform"]:
        if hasattr(cls, method_name):
            attrs[method_name] = get_func_attrs(getattr(cls, method_name))
    return attrs


before = [get_fit_attrs(c) for _, c in sklearn.utils.all_estimators()]
mlflow.sklearn.autolog()
after = [get_fit_attrs(c) for _, c in sklearn.utils.all_estimators()]

for b, a in zip(before, after):
    assert b == a
