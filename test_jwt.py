import requests
import uuid

# Login
resp = requests.post(
    "http://localhost:5000/api/2.0/mlflow/auth/login",
    json={"email": "user_a@test.com", "password": "pass_a"},
)
resp.raise_for_status()
access_token = resp.json()["access_token"]

# Create an experiment
name = uuid.uuid4().hex
resp = requests.post(
    "http://localhost:5000/api/2.0/mlflow/experiments/create",
    json={"name": name},
)

assert resp.status_code == 401

resp = requests.post(
    "http://localhost:5000/api/2.0/mlflow/experiments/create",
    json={"name": name},
    headers={"Authorization": f"Bearer {access_token}"},
)
resp.raise_for_status()
data = resp.json()
experiment_id = data["experiment_id"]

resp = requests.get(
    f"http://localhost:5000/api/2.0/mlflow/auth/experiments/{experiment_id}/permissions",
    headers={"Authorization": f"Bearer {access_token}"},
)
resp.raise_for_status()
data = resp.json()
print(data)
