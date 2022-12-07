from mlflow.pyfunc import PythonModel
import pandas as pd
import numpy as np


class WrappedRecipeModel(PythonModel):
    def __init__(self, classifier, predict_scores_for_all_classes=True, prefix="predicted_"):
        super().__init__()
        self._classifier = classifier
        self.predict_scores_for_all_classes = predict_scores_for_all_classes
        self.prefix = prefix
        self.classification = False

    def predict(self, model_input):
        predicted_label = self._classifier.predict(model_input)
        if not hasattr(self._classifier, "classes_"):
            return predicted_label

        self.classification = True
        classes = self._classifier.classes_
        score_cols = [f"{self.prefix}score_" + str(c) for c in classes]
        output = {}
        try:
            probabilities = self._classifier.predict_proba(model_input)
            if self.predict_scores_for_all_classes:
                output = pd.DataFrame(columns=score_cols, data=probabilities)
            output[f"{self.prefix}score"] = np.max(probabilities, axis=1)
            output[f"{self.prefix}label"] = predicted_label

            return output
        except Exception:
            # swallow that some models don't have predict_proba.
            self.classification = False
            return predicted_label
