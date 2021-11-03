# Importing Libraries
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature


# Loading model
infile = open('heart-attack-prediction.pkl','rb')
model = pickle.load(infile)
print ('Loaded Model')
infile.close()

# Creating Signature
mlflow.set_tracking_uri("sqlite:///mlruns.db")

input_schema = Schema([
  ColSpec("double", "age"),
  ColSpec("integer", "sex"),
  ColSpec("integer", "cp"),
  ColSpec("double", "trtbps"),
  ColSpec("double", "chol"),
  ColSpec("integer", "fbs"),
  ColSpec("integer", "restecg"),
  ColSpec("double", "thalachh"),
  ColSpec("integer", "exng"),
  ColSpec("double", "oldpeak"),
  ColSpec("integer", "slp"),
  ColSpec("integer", "caa"),
  ColSpec("integer", "thall"),
])

signature = ModelSignature(inputs=input_schema)


#Logging Model artifacts
mlflow.sklearn.log_model(model, "heart-attack-prediction-model",registered_model_name="heart-attack-prediction-model",signature=signature)
print ('Logged another model')

#Transition into production
client = MlflowClient()
client.transition_model_version_stage(
    name="heart-attack-prediction-model",
    version=1,
    stage="Production"
    )
