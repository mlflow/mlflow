"""
THE 'mlflow.evaluation` MODULE IS LEGACY AND WILL BE REMOVED SOON. PLEASE DO NOT USE THESE CLASSES
IN NEW CODE. INSTEAD, USE `mlflow/entities/assessment.py` FOR ASSESSMENT CLASSES.
"""

from mlflow.evaluation.assessment import Assessment, AssessmentSource, AssessmentSourceType
from mlflow.evaluation.evaluation import Evaluation
from mlflow.evaluation.fluent import log_evaluations

__all__ = [
    "Assessment",
    "AssessmentSource",
    "AssessmentSourceType",
    "Evaluation",
    "log_evaluations",
]
