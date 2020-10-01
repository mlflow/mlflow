from mlflow.utils.annotations import experimental


@experimental
def log_explanation(predict_func, features, explanation_path=None):
    """
    Generate a SHAP explanation and log it.

    :param predict_func: A function to compute the output of a model
                         (e.g. ``predict`` method of scikit-learn regressors).
    :type predict_func: function
    :param features: A matrix of features on which to explain the model's output.
    :type features: np.ndarray or pd.dataframe
    :param explanation_path: Run-relative artifact path the explanation is saved to.
                             If unspecified, defaults to ``"shap"``.
    :type explanation_path: str

    :return: URI of the logged SHAP explanation
    :rtype: str

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

    .. image:: ../_static/images/shap-ui-screenshot.png
        :width: 900
    """
    raise NotImplementedError("Not implemented yet")
