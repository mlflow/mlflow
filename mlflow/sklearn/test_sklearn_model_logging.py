import pytest 
from sklearn.dummy import DummyClassifier
from mlflow.sklearn import log_model

def test_artifact_path_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="'artifact_path' is deprecated and will be removed in a future release. Please use 'name' instead."):
        log_model(DummyClassifier, artifact_path="some_path")