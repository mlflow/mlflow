import matplotlib.pyplot as plt
import pandas as pd


def to_pandas_Xy(dataset):
    """
    Extracts `data` and `target` from a scikit-learn dataset and returns them as a pandas DataFrame
    and Series.
    """
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y
