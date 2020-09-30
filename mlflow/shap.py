from mlflow.utils.annotations import experimental


@experimental
def log_explanation(predict_func, features, explanation_path=None):
    """
    Generate a SHAP explanation and log it.

    Parameters
    ----------
    predict_func : function
        A function to compute the output of the model you'd like to explain
        (e.g. ``predict`` method of scikit-learn regressors).
    features : np.ndarray or pd.dataframe
        A matrix of features on which to explain the model's output.
    explanation_path : str
        Run-relative artifact path the explanation is saved to. If unspecified,
        defaults to ``"shap"``.

    Returns
    -------
    explation_uri : str
        URI of the logged SHAP explanation


    .. code-block:: python
        :caption: Example

        import pandas as pd
        from sklearn.datasets import load_boston
        from sklearn.linear_model import LinearRegression

        import mlflow

        # prepare training data
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = boston.target

        # train a model
        model = LinearRegression()
        model.fit(X, y)

        # log a SHAP explanation
        mlflow.shap.log_explanation(model.predict, X)
    """
    raise NotImplementedError("Not implemented yet")
