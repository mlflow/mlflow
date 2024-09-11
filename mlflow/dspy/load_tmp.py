import mlflow

model_uri = "runs:/0c4520ecb8c446a78859474178c0bfa9/model"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("test-dspy-logging")

loaded_model = mlflow.pyfunc.load_model(model_uri)

import pdb

pdb.set_trace()

loaded_model.predict("What is 2 + 2?")
