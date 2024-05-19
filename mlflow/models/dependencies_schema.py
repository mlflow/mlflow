from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from mlflow.utils.annotations import experimental

DATABRICKS_RETRIEVER_PRIMARY_KEY = "__databricks_retriever_primary_key__"
DATABRICKS_RETRIEVER_TEXT_COLUMN = "__databricks_retriever_text_column__"
DATABRICKS_RETRIEVER_DOC_URI = "__databricks_retriever_doc_uri__"
DATABRICKS_RETRIEVER_OTHER_COLUMNS = "__databricks_retriever_other_columns__"


class DependenciesSchemasType(Enum):
    """
    Enum to define the different types of dependencies schemas for the model.
    """

    RETRIEVERS = "retrievers"


@experimental
def set_retriever_schema(
    primary_key: str,
    text_column: str,
    doc_uri: Optional[str] = None,
    other_columns: Optional[List[str]] = None,
):
    """
    After defining your vector store in a Python file or notebook, call
    set_retriever_schema() so that, when MLflow retrieves documents during
    model inference, MLflow can interpret the fields in each retrieved document and
    determine which fields correspond to the document text, document URI, etc.

    Args:
        primary_key: The primary key of the vector index.
        text_column: The name of the text column to use for the embeddings.
        doc_uri: The name of the column that contains the document URI.
        other_columns: A list of other columns that are part of the vector index
                          that need to be retrieved during trace logging.
        Note: Make sure the text column specified is in the index.

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
    globals()[DATABRICKS_RETRIEVER_PRIMARY_KEY] = primary_key
    globals()[DATABRICKS_RETRIEVER_TEXT_COLUMN] = text_column
    globals()[DATABRICKS_RETRIEVER_DOC_URI] = doc_uri
    globals()[DATABRICKS_RETRIEVER_OTHER_COLUMNS] = other_columns or []


def _get_retriever_schema():
    """
    Get the vector search schema defined by the user.

    Returns:
        VectorSearchIndex: The vector search index schema.
    """
    if not globals().get(DATABRICKS_RETRIEVER_PRIMARY_KEY, None) or not globals().get(
        DATABRICKS_RETRIEVER_TEXT_COLUMN, None
    ):
        return []

    return [
        RetrieverSchema(
            name="retriever",
            primary_key=globals().get(DATABRICKS_RETRIEVER_PRIMARY_KEY, None),
            text_column=globals().get(DATABRICKS_RETRIEVER_TEXT_COLUMN, None),
            doc_uri=globals().get(DATABRICKS_RETRIEVER_DOC_URI, None),
            other_columns=globals().get(DATABRICKS_RETRIEVER_OTHER_COLUMNS, None),
        )
    ]


def _clear_retriever_schema():
    """
    Clear the vector search schema defined by the user.
    """
    globals().pop(DATABRICKS_RETRIEVER_PRIMARY_KEY, None)
    globals().pop(DATABRICKS_RETRIEVER_TEXT_COLUMN, None)
    globals().pop(DATABRICKS_RETRIEVER_DOC_URI, None)
    globals().pop(DATABRICKS_RETRIEVER_OTHER_COLUMNS, None)


def _clear_dependencies_schema():
    """
    Clear all the dependencies schema defined by the user.
    """
    # Clear the vector search schema
    _clear_retriever_schema()


@contextmanager
def _get_dependencies_schema():
    dependencies_schema = DependenciesSchemas(retriever_schemas=_get_retriever_schema())
    try:
        yield dependencies_schema
    finally:
        _clear_dependencies_schema()


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
    def from_dict(cls, data: Dict[str, str]):
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
        other_columns: Optional[List[str]] = None,
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
    def from_dict(cls, data: Dict[str, str]):
        return cls(
            name=data["name"],
            primary_key=data["primary_key"],
            text_column=data["text_column"],
            doc_uri=data.get("doc_uri"),
            other_columns=data.get("other_columns", []),
        )


@dataclass
class DependenciesSchemas:
    retriever_schemas: List[RetrieverSchema] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Dict[DependenciesSchemasType, List[Dict]]]:
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
