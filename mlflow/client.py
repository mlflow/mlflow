"""
The ``mlflow.client`` module provides a Python CRUD interface to MLflow Experiments, Runs,
Model Versions, and Registered Models. This is a lower level API that directly translates to MLflow
`REST API <../rest-api.html>`_ calls.
For a higher level API for managing an "active run", use the :py:mod:`mlflow` module.
"""

from mlflow.tracking.client import MlflowClient
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.requests import Request
from mlflow import MlflowClient as mlflow_client
import os

app = FastAPI()

@app.put("/api/2.0/mlflow-artifacts/artifacts/{path}")
async def upload_artifact(path: str, file: bytes = File(...)):
    mlflow_client().upload_artifact(path, file)
    return {"message": "Artifact uploaded successfully"}

@app.get("/api/2.0/mlflow-artifacts/artifacts/{path}")
async def download_artifact(path: str, request: Request):
    artifact_path = mlflow_client().get_artifact_path(path)
    if os.path.exists(artifact_path):
        return FileResponse(artifact_path)
    else:
        return {"message": "Artifact not found"}