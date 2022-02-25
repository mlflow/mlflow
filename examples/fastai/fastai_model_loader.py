import os
import numpy as np
import pandas as pd

class _FastaiModelWrapperPatch:
    def __init__(self, learner):
        self.learner = learner

    def predict(self, dataframe):
        dl = self.learner.dls.test_dl(dataframe)
        preds, _ = self.learner.get_preds(dl=dl)
        return pd.Series(map(np.array, preds.numpy())).to_frame('predictions')

def _load_model(path):
    from fastai.learner import load_learner

    return load_learner(os.path.abspath(path))

def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    :param path: Local filesystem path to the MLflow Model with the ``fastai`` flavor.
    """
    print("Model loaded from:",path)
    return _FastaiModelWrapperPatch(_load_model(path))