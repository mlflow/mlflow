import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
import numpy as np


class WrappedRecipeEstimator:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def fit(self, x, Y, *args, **kwargs):
        print("IN FIT Y", self.label_encoder.transform(Y))
        print("IN FIT Y_TRANSFORM", self.label_encoder.transform(Y))
        return self.model.fit(x, self.label_encoder.transform(Y), *args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
