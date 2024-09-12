import mlflow

model_uri = "runs:/7bfc9bf087b34b3da93f27f03d085d83/model"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test-dspy-logging")

loaded_model = mlflow.pyfunc.load_model(model_uri)
import pdb

pdb.set_trace()

print(loaded_model.predict("What is 2 + 2?"))
