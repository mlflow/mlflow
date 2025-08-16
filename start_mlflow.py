import os
import subprocess

backend_uri = os.environ.get("MLFLOW_BACKEND_STORE_URI", "/home/mlflow/mlruns")
artifact_root = os.environ.get("MLFLOW_DEFAULT_ARTIFACT_ROOT", "/home/mlflow/mlruns")

# --- Enable DEBUG on both MLflow and Gunicorn ---
os.environ["MLFLOW_LOGGING_LEVEL"]    = "DEBUG"            # :contentReference[oaicite:2]{index=2}
os.environ["MLFLOW_GUNICORN_OPTS"]    = "--log-level debug" # :contentReference[oaicite:3]{index=3}


print("Starting MLflow Tracking Server...")
print(f"Backend URI: {backend_uri}")
print(f"Artifact Root: {artifact_root}")

cmd = [
    "mlflow", "server",
    "--backend-store-uri", backend_uri,
    "--default-artifact-root", artifact_root,
    "--host", "0.0.0.0",
    "--port", "8080"
]

subprocess.run(cmd)
