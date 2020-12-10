import mlflow
import shap
import xgboost
import pickle
import numpy as np
import sklearn
from mlflow.utils.environment import _mlflow_conda_env


def get_sklearn_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    pip_deps = ["shap=={}".format(shap.__version__)]

    return _mlflow_conda_env(
        additional_conda_deps=["scikit-learn={}".format(sklearn.__version__)],
        additional_pip_deps=pip_deps,
        additional_conda_channels=None,
    )

def test_basic_log_explainer():

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
    
        explainer_original = shap.Explainer(model.predict, X, algorithm='permutation')
        shap_values_original = explainer_original(X[:5])

        mlflow.shap.log_explainer(explainer_original, "test_explainer", conda_env = get_sklearn_conda_env())

        explainer_new = mlflow.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_new(X[:5])

        assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
        assert type(explainer_original) == type(explainer_new)
        assert type(explainer_original.masker) == type(explainer_new.masker)


def test_basic_log_explainer_pyfunc():

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
    
        explainer_original = shap.Explainer(model.predict, X, algorithm='permutation')
        shap_values_original = explainer_original(X[:1])

        mlflow.shap.log_explainer(explainer_original, "test_explainer", conda_env = get_sklearn_conda_env())

        explainer_pyfunc = mlflow.pyfunc.load_model("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_pyfunc.predict(X[:1])
        
        assert shap_values_original.shape == shap_values_new.shape