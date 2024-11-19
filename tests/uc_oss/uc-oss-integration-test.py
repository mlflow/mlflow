import mlflow
import os
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

catalog = "unity"
schema = "default"
registered_model_name = "iris"
model_name = f"{catalog}.{schema}.{registered_model_name}"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("uc:http://127.0.0.1:8080")
mlflow.set_experiment("iris-uc-oss")

# COMMAND ----------

try:
  mlflow.MlflowClient().get_registered_model(model_name)
except Exception as e:
  e.args[0].startswith("NOT_FOUND")
  pass
else:
  assert False, "Expected exception not raised"

# COMMAND ----------

X, y = datasets.load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def build_model():
    with mlflow.start_run():
        # Train a sklearn model on the iris dataset
        clf = RandomForestClassifier(max_depth=7)
        clf.fit(X_train, y_train)
        # Take the first row of the training dataset as the model input example.
        input_example = X.iloc[[0]]
        # Log the model and register it as a new version in UC.
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            # The signature is automatically inferred from the input example and its predicted output.
            input_example=input_example,
            registered_model_name=model_name,
        )

# COMMAND ----------

try:
    build_model()
    pass
except Exception as e:
    assert False

# COMMAND ----------

model_version = 1
model_uri = f"models:/{model_name}/{model_version}"
rm_desc = "UC-OSS/MLflow Iris model"
mv_desc = "Version 1 of the UC-OSS/MLflow Iris model"

# Load the model and do some batch inference.
try:
    # By specifying the UC OSS model uri, mlflow will make UC OSS
    # REST API calls to retrieve the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    predictions = loaded_model.predict(X_test)
    iris_feature_names = datasets.load_iris().feature_names
    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    result[:4]
    pass
except Exception as e:
    assert False

# COMMAND ----------

try:
    # list_artifacts will use the UC OSS model URI and make REST API calls to
    # UC OSS to:
    #   1) retrieve credentials (none for file based UC OSS)
    #   2) use the storage location returned from UC OSS for the model version
    #      list the artifacts stored in the location
    mlflow.artifacts.list_artifacts(model_uri)
    pass
except Exception as e:
    assert False

# COMMAND ----------

path = os.path.join("/tmp", "models", model_name, str(model_version))
print(path)

try:
    # download_artifacts will use the UC OSS model URI and make REST API calls
    # to UC OSS to:
    #   1) retrieve credentials (none for file based UC OSS)
    #   2) copy the artifact files from the storage location to the
    #      destintation path
    mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{model_name}/{model_version}",
    dst_path=path,
    )
    os.system(f"cat {path}/requirements.txt")
    pass
except Exception as e:
    assert False

# COMMAND ----------

# Test get RM/MV works
model1 = mlflow.MlflowClient().get_registered_model(model_name)
assert model1.name == model_name
assert model1.description == ""
model_v1 = mlflow.MlflowClient().get_model_version(name=model_name, version=model_version)
assert model_v1.name == model_name
assert model_v1.version == 1
assert model_v1.description == ""

# Test update RM/MV works
mlflow.MlflowClient().update_registered_model(model_name, description=rm_desc)
model2 = mlflow.MlflowClient().get_registered_model(model_name)
assert model2.name == model_name
assert model2.description == rm_desc
mlflow.MlflowClient().update_model_version(name=model_name, version=model_version, description=mv_desc)
model_v1_2 = mlflow.MlflowClient().get_model_version(name=model_name, version=model_version)
assert model_v1_2.name == model_name
assert model_v1_2.version == 1
assert model_v1_2.description == mv_desc

rms = mlflow.MlflowClient().search_registered_models()
total_rms = len(rms)
assert len(rms) > 1
mvs = mlflow.MlflowClient().search_model_versions(f"name='{model_name}'")
assert len(mvs) == 1
mlflow.MlflowClient().delete_model_version(name=model_name, version=1)
mvs = mlflow.MlflowClient().search_model_versions(f"name='{model_name}'")
assert len(mvs) == 0
mlflow.MlflowClient().delete_registered_model(name=model_name)
rms = mlflow.MlflowClient().search_registered_models()
try:
    mlflow.MlflowClient().get_registered_model(model_name)
except Exception as e:
    e.args[0].startswith("NOT_FOUND")
    pass
else:
    assert False, "Expected exception not raised"
