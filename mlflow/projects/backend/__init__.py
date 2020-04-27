"""
This module defines developer APIs for defining pluggable execution backends
for MLflow projects. See `MLflow Plugins <../../plugins.html>`_ for more information.
"""
from mlflow.projects.backend.abstract_backend import AbstractBackend

__all__ = ["AbstractBackend"]
