import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
import numpy as np


class WrappedRecipeModel(PythonModel):
    def __init__(self, predict_scores_for_all_classes=True, prefix="predicted_"):
        super().__init__()
        self.predict_scores_for_all_classes = predict_scores_for_all_classes
        self.prefix = prefix

    def load_context(self, context):
        self._classifier = mlflow.sklearn.load_model(context.artifacts["model_path"])

    def predict(self, context, model_input):
        predicted_label = self._classifier.predict(model_input)
        # Only classification recipe would be have multiple classes in the target column
        # So if it doesn't have multiple classes, return back the predicted_label
        # or else we try to commute the predict_proba.
        if not hasattr(self._classifier, "classes_"):
            return predicted_label

        classes = self._classifier.classes_
        score_cols = [f"{self.prefix}score_" + str(c) for c in classes]
        output = {}
        if hasattr(self._classifier, "predict_proba"):
            probabilities = self._classifier.predict_proba(model_input)
            if self.predict_scores_for_all_classes:
                output = pd.DataFrame(columns=score_cols, data=probabilities)
            output[f"{self.prefix}score"] = np.max(probabilities, axis=1)
            output[f"{self.prefix}label"] = predicted_label

            return output
        else:
            return predicted_label
