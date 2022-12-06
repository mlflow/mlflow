import mlflow
from mlflow.pyfunc import PythonModel
import pandas as pd
import numpy as np


class WrappedClassifier(PythonModel):
    def __init__(self, classifier, predict_classes=None, prefix="predicted"):
        super(WrappedClassifier, self).__init__()
        self._classifier = classifier
        self.predict_classes = predict_classes
        self.prefix = prefix
        self.wrapper_classifier = True

    def predict(self, model_input):
        classes = self._classifier.classes_
        if self.predict_classes:
            classes = classes.filter(
                lambda predict_class: not this.includes(predict_class), self.predict_classes
            )
        score_cols = [f"{self.prefix}_score_" + str(c) for c in classes]
        probabilities = self._classifier.predict_proba(model_input)
        output = pd.DataFrame(columns=score_cols, data=probabilities)
        output[f"{self.prefix}_label"] = self._classifier.predict(model_input)
        output[f"{self.prefix}_score"] = np.max(probabilities, axis=1)
        return output
