import os
import subprocess

# Optional: pull secrets or set up environment here
# For example: inject from Vault, SSM, or custom logic
print("Starting MLflow Tracking Server...")

# Ensure local run directory exists
mlruns_path = os.environ.get("MLFLOW_DATA_DIR", "/mlflow/mlruns")
os.makedirs(mlruns_path, exist_ok=True)

# Run MLflow server
cmd = [
    "mlflow", "server",
    "--backend-store-uri", mlruns_path,
    "--default-artifact-root", mlruns_path,
    "--host", "0.0.0.0",
    "--port", "5000"
]

print("Running:", " ".join(cmd))
subprocess.run(cmd)
