"""
The :py:mod:`mlflow.types` module defines data types and utilities to be used by other mlflow
components to describe interface independent of other frameworks or languages.
"""

from mlflow.version import IS_TRACING_SDK_ONLY

if not IS_TRACING_SDK_ONLY:
    try:
        import mlflow.types.llm  # noqa: F401

        # Our typing system depends on numpy, which is not included in mlflow-tracing package
        # or environments where numpy is not installed (e.g. mlflow-skinny without numpy).
        from mlflow.types.schema import ColSpec, DataType, ParamSchema, ParamSpec, Schema, TensorSpec

        __all__ = [
            "Schema",
            "ColSpec",
            "DataType",
            "TensorSpec",
            "ParamSchema",
            "ParamSpec",
        ]
    except ImportError:
        pass
