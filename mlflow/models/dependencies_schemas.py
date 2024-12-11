from __future__ import annotations

import logging
import threading
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.models.model import Model

_logger = logging.getLogger(__name__)


class DependenciesSchemasType(str, Enum):
    """
    Enum to define the different types of dependencies schemas for the model.
    """

    RETRIEVERS = "retrievers"


# Key to store the dependencies schemas in the model metadata
DEPENDENCIES_SCHEMA_KEY = "dependencies_schemas"

# Global variable to store the dependencies schemas defined by the user
_DEPENDENCIES_SCHEMAS: dict[DependenciesSchemasType, list[Schema]] = {}

# Lock to ensure thread safety when updating the dependencies schemas
_DEPENDENCIES_SCHEMAS_LOCK = threading.RLock()


def _synchronized(f):
    def wrapper(*args, **kwargs):
        with _DEPENDENCIES_SCHEMAS_LOCK:
            return f(*args, **kwargs)

    return wrapper


@experimental
@_synchronized
def set_retriever_schema(
    *,
    primary_key: str,
    text_column: str,
    doc_uri: Optional[str] = None,
    other_columns: Optional[list[str]] = None,
    name: Optional[str] = "retriever",
):
    """
    After defining your vector store in a Python file or notebook, call
    set_retriever_schema() so that, when MLflow retrieves documents during
    model inference, MLflow can interpret the fields in each retrieved document and
    determine which fields correspond to the document text, document URI, etc.

    Args:
        primary_key: The primary key of the retriever or vector index.
        text_column: The name of the text column to use for the embeddings.
        doc_uri: The name of the column that contains the document URI.
        other_columns: A list of other columns that are part of the vector index
                          that need to be retrieved during trace logging.
        name: The name of the retriever tool or vector store index.

    .. code-block:: Python
            :caption: Example

            from mlflow.models import set_retriever_schema

            set_retriever_schema(
                primary_key="chunk_id",
                text_column="chunk_text",
                doc_uri="doc_uri",
                other_columns=["title"],
            )
    """
    retriever_schemas = _DEPENDENCIES_SCHEMAS.get(DependenciesSchemasType.RETRIEVERS.value, [])

    schema = RetrieverSchema(
        name=name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        other_columns=other_columns or [],
    )

    # Check if a retriever schema with the same name already exists
    if existing_schema := next((sc for sc in retriever_schemas if sc.name == name), None):
        if existing_schema == schema:
            # No difference, no need to warn or update
            return
        else:
            # Differences found, issue a warning
            _logger.warning(
                f"A retriever schema with the name '{name}' already exists. "
                "Overriding the existing schema."
            )
            # Override the fields of the existing schema
            existing_schema.primary_key = primary_key
            existing_schema.text_column = text_column
            existing_schema.doc_uri = doc_uri
            existing_schema.other_columns = other_columns or []
    else:
        retriever_schemas.append(schema)

    _DEPENDENCIES_SCHEMAS[DependenciesSchemasType.RETRIEVERS.value] = retriever_schemas


def get_dependencies_schemas():
    """
    Get the dependencies schemas defined by the user.

    Returns:
        dict: A dictionary containing the dependencies schemas.
    """
    return _DEPENDENCIES_SCHEMAS.copy()


@_synchronized
def clear_dependencies_schemas():
    """
    Clear all the dependencies schema defined by the user.
    """
    _DEPENDENCIES_SCHEMAS.clear()


@_synchronized
def set_dependencies_schema_to_model(
    model: "Model", schema: dict[DependenciesSchemasType, list[Schema]]
) -> dict:
    """
    Set the dependencies schema to the logged model metadata.

    Returns:
        dict: A dictionary containing the dependencies schemas.
    """
    if not _DEPENDENCIES_SCHEMAS:
        return

    dependencies_schemas = {k: [v.__dict__ for v in vs] for k, vs in _DEPENDENCIES_SCHEMAS.items()}
    if model.metadata is None:
        model.metadata = {}

    # TODO: define a constant for the key
    model.metadata["dependencies_schema"] = dependencies_schemas


@dataclass
class Schema(ABC):
    """Base class for defining the resources needed to serve a model."""


@dataclass
class RetrieverSchema(Schema):
    """
    Define vector search index resource to serve a model.

    Args:
        name (str): The name of the vector search index schema.
        primary_key (str): The primary key for the index.
        text_column (str): The main text column for the index.
        doc_uri (Optional[str]): The document URI for the index.
        other_columns (Optional[List[str]]): Additional columns in the index.
    """

    name: str
    primary_key: str
    text_column: str
    doc_uri: Optional[str] = None
    other_columns: Optional[list[str]] = None
