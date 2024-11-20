import json
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.models.model import Model

_logger = logging.getLogger(__name__)


class DependenciesSchemasType(Enum):
    """
    Enum to define the different types of dependencies schemas for the model.
    """

    RETRIEVERS = "retrievers"


@experimental
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
    retriever_schemas = globals().get(DependenciesSchemasType.RETRIEVERS.value, [])

    # Check if a retriever schema with the same name already exists
    existing_schema = next((schema for schema in retriever_schemas if schema["name"] == name), None)

    if existing_schema is not None:
        # Compare all relevant fields
        if (
            existing_schema["primary_key"] == primary_key
            and existing_schema["text_column"] == text_column
            and existing_schema["doc_uri"] == doc_uri
            and existing_schema["other_columns"] == (other_columns or [])
        ):
            # No difference, no need to warn or update
            return
        else:
            # Differences found, issue a warning
            _logger.warning(
                f"A retriever schema with the name '{name}' already exists. "
                "Overriding the existing schema."
            )
            # Override the fields of the existing schema
            existing_schema["primary_key"] = primary_key
            existing_schema["text_column"] = text_column
            existing_schema["doc_uri"] = doc_uri
            existing_schema["other_columns"] = other_columns or []
    else:
        retriever_schemas.append(
            {
                "primary_key": primary_key,
                "text_column": text_column,
                "doc_uri": doc_uri,
                "other_columns": other_columns or [],
                "name": name,
            }
        )

    globals()[DependenciesSchemasType.RETRIEVERS.value] = retriever_schemas


def _get_retriever_schema():
    """
    Get the vector search schema defined by the user.

    Returns:
        VectorSearchIndex: The vector search index schema.
    """
    retriever_schemas = globals().get(DependenciesSchemasType.RETRIEVERS.value, [])
    if not retriever_schemas:
        return []

    return [
        RetrieverSchema(
            name=retriever.get("name"),
            primary_key=retriever.get("primary_key"),
            text_column=retriever.get("text_column"),
            doc_uri=retriever.get("doc_uri"),
            other_columns=retriever.get("other_columns"),
        )
        for retriever in retriever_schemas
    ]


def _clear_retriever_schema():
    """
    Clear the vector search schema defined by the user.
    """
    globals().pop(DependenciesSchemasType.RETRIEVERS.value, None)


def _clear_dependencies_schemas():
    """
    Clear all the dependencies schema defined by the user.
    """
    # Clear the vector search schema
    _clear_retriever_schema()


@contextmanager
def _get_dependencies_schemas():
    dependencies_schemas = DependenciesSchemas(retriever_schemas=_get_retriever_schema())
    try:
        yield dependencies_schemas
    finally:
        _clear_dependencies_schemas()


def _get_dependencies_schema_from_model(model: "Model") -> Optional[dict]:
    """
    Get the dependencies schema from the logged model metadata.

    `dependencies_schemas` is a dictionary that defines the dependencies schemas, such as
    the retriever schemas. This code is now only useful for Databricks integration.
    """
    if model.metadata and "dependencies_schemas" in model.metadata:
        dependencies_schemas = model.metadata["dependencies_schemas"]
        return {
            "dependencies_schemas": {
                dependency: json.dumps(schema)
                for dependency, schema in dependencies_schemas.items()
            }
        }
    return None


@dataclass
class Schema(ABC):
    """
    Base class for defining the resources needed to serve a model.

    Args:
        type (ResourceType): The type of the schema.
    """

    type: DependenciesSchemasType

    @abstractmethod
    def to_dict(self):
        """
        Convert the resource to a dictionary.
        Subclasses must implement this method.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, str]):
        """
        Convert the dictionary to a Resource.
        Subclasses must implement this method.
        """


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

    def __init__(
        self,
        name: str,
        primary_key: str,
        text_column: str,
        doc_uri: Optional[str] = None,
        other_columns: Optional[list[str]] = None,
    ):
        super().__init__(type=DependenciesSchemasType.RETRIEVERS)
        self.name = name
        self.primary_key = primary_key
        self.text_column = text_column
        self.doc_uri = doc_uri
        self.other_columns = other_columns or []

    def to_dict(self):
        return {
            self.type.value: [
                {
                    "name": self.name,
                    "primary_key": self.primary_key,
                    "text_column": self.text_column,
                    "doc_uri": self.doc_uri,
                    "other_columns": self.other_columns,
                }
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(
            name=data["name"],
            primary_key=data["primary_key"],
            text_column=data["text_column"],
            doc_uri=data.get("doc_uri"),
            other_columns=data.get("other_columns", []),
        )


@dataclass
class DependenciesSchemas:
    retriever_schemas: list[RetrieverSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, dict[DependenciesSchemasType, list[dict]]]:
        if not self.retriever_schemas:
            return None

        return {
            "dependencies_schemas": {
                DependenciesSchemasType.RETRIEVERS.value: [
                    index.to_dict()[DependenciesSchemasType.RETRIEVERS.value][0]
                    for index in self.retriever_schemas
                ],
            }
        }
