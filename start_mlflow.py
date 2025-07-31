import os
import subprocess

backend_uri = os.environ.get("MLFLOW_BACKEND_STORE_URI", "/home/mlflow/mlruns")
artifact_root = os.environ.get("MLFLOW_DEFAULT_ARTIFACT_ROOT", "/home/mlflow/mlruns")

print("Starting MLflow Tracking Server...")
print(f"Backend URI: {backend_uri}")
print(f"Artifact Root: {artifact_root}")

cmd = [
    "mlflow", "server",
    "--backend-store-uri", backend_uri,
    "--default-artifact-root", artifact_root,
    "--host", "0.0.0.0",
    "--port", "5000"
]

subprocess.run(cmd)
