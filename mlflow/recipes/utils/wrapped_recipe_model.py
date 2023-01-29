import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
import numpy as np


class WrappedRecipeModel(PythonModel):
    def __init__(
        self, predict_scores_for_all_classes, predict_prefix, target_column_class_labels=None
    ):
        super().__init__()
        self.predict_scores_for_all_classes = predict_scores_for_all_classes
        self.predict_prefix = predict_prefix
        self.target_column_class_labels = target_column_class_labels

    def load_context(self, context):
        self._classifier = mlflow.sklearn.load_model(context.artifacts["model_path"])

    def predict(self, context, model_input):
        predicted_label = self._classifier.predict(model_input)
        # Only classification recipe would be have multiple classes in the target column
        # So if it doesn't have multiple classes, return back the predicted_label
        # or else we try to commute the predict_proba if the algorithm supports it.
        if (
            not hasattr(self._classifier, "classes_")
            or not hasattr(self._classifier, "predict_proba")
            or not self.predict_scores_for_all_classes
        ):
            return predicted_label

        classes = (
            self.target_column_class_labels
            if self.target_column_class_labels is not None
            else self._classifier.classes_
        )
        score_cols = [f"{self.predict_prefix}score_" + str(c) for c in classes]
        probabilities = self._classifier.predict_proba(model_input)
        output = pd.DataFrame(columns=score_cols, data=probabilities)
        output[f"{self.predict_prefix}score"] = np.max(probabilities, axis=1)
        output[f"{self.predict_prefix}label"] = predicted_label

        return output
