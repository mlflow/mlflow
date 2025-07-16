"""
The :py:mod:`mlflow.types` module defines data types and utilities to be used by other mlflow
components to describe interface independent of other frameworks or languages.
"""

from mlflow.version import IS_FULL_MLFLOW

if IS_FULL_MLFLOW:
    import mlflow.types.llm  # noqa: F401

    # Our typing system depends on numpy, which is not included in mlflow-tracing package
    from mlflow.types.schema import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec

    __all__ = [
        "Schema",
        "ColSpec",
        "DataType",
        "TensorSpec",
        "ParamSchema",
        "ParamSpec",
    ]
