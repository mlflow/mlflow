"""
The :py:mod:`mlflow.types` module defines mlflow data types and provides utilities such as type
inference from python types.

The types defined here can be used by other mlflow components to describe interface independent of
other frameworks or languages.  The data types can be organized in Schema that declares a sequence
of optionally named typed columns.
"""

from .schema import DataType, ColSpec, Schema

__all__ = ["DataType", "ColSpec", "Schema"]

