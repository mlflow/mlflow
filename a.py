import time
import mlflow
import uuid

client = mlflow.MlflowClient()

for i in range(10):
    print("Before", int(time.time() * 1000))
    client.create_experiment(uuid.uuid4().hex)
    print("After", int(time.time() * 1000))
