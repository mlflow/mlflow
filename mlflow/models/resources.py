from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List


class ResourceType(Enum):
    DATABRICKS_VECTOR_SEARCH_INDEX_NAME = "databricks_vector_search_index_name"
    DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME = "databricks_vector_search_endpoint_name"
    DATABRICKS_EMBEDDINGS_ENDPOINT_NAME = "databricks_embeddings_endpoint_name"
    DATABRICKS_LLM_ENDPOINT_NAME = "databricks_llm_endpoint_name"
    DATABRICKS_CHAT_ENDPOINT_NAME = "databricks_chat_endpoint_name"


class Providers(Enum):
    DATABRICKS = "databricks"


@dataclass
class Resource(ABC):
    """
    Base class for all resources to define the resources needed to serve a model.

    Args:
        type (ResourceType): The resource type.
        target_uri (str): The target URI where these resources are hosted.
    """

    type: ResourceType
    target_uri: Providers

    @abstractmethod
    def to_dict(self):
        """
        Convert the resource to a dictionary.
        Subclasses must implement this method.
        """


@dataclass
class DatabricksResource(Resource, ABC):
    """
    Base class to define all the Databricks resources to serve a model.
    """

    target_uri: Providers = Providers.DATABRICKS.value


@dataclass
class DatabricksLLMEndpoint(DatabricksResource):
    """
    Define Databricks LLM endpoint resource to serve a model.

    Args:
        endpoint_name (str): The name of all the databricks endpoints used by the model.
    """

    type: ResourceType = ResourceType.DATABRICKS_LLM_ENDPOINT_NAME
    endpoint_name: str = None

    def to_dict(self):
        return {self.type.value: [self.endpoint_name]} if self.endpoint_name else {}


@dataclass
class DatabricksChatEndpoint(DatabricksResource):
    """
    Define Databricks LLM endpoint resource to serve a model.

    Args:
        endpoint_name (str): The name of all the databricks chat endpoints
        used by the model.
    """

    type: ResourceType = ResourceType.DATABRICKS_CHAT_ENDPOINT_NAME
    endpoint_name: str = None

    def to_dict(self):
        return {self.type.value: [self.endpoint_name]} if self.endpoint_name else {}


@dataclass
class DatabricksEmbeddingEndpoint(DatabricksResource):
    """
    Define Databricks embedding endpoint resource to serve a model.

    Args:
        endpoint_name (str): The name of all the databricks embedding endpoints
        used by the model.
    """

    type: ResourceType = ResourceType.DATABRICKS_EMBEDDINGS_ENDPOINT_NAME
    endpoint_name: str = None

    def to_dict(self):
        return {self.type.value: [self.endpoint_name]} if self.endpoint_name else {}


@dataclass
class DatabricksVectorSearchEndpoint(DatabricksResource):
    """
    Define Databricks vector search endpoint resource to serve a model.

    Args:
        endpoint_name (str): The name of all the databricks vector search endpoints
        used by the model.
    """

    type: ResourceType = ResourceType.DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME
    endpoint_name: str = None

    def to_dict(self):
        return {self.type.value: [self.endpoint_name]} if self.endpoint_name else {}


@dataclass
class DatabricksVectorSearchIndexName(DatabricksResource):
    """
    Define Databricks vector search index name resource to serve a model.

    Args:
        index_name (str): The name of all the databricks vector search index names
        used by the model.
    """

    type: ResourceType = ResourceType.DATABRICKS_VECTOR_SEARCH_INDEX_NAME
    index_name: str = None

    def to_dict(self):
        return {self.type.value: [self.index_name]} if self.index_name else {}


class _ResourceBuilder:
    """
    Private builder class to build the resources dictionary.
    """

    @staticmethod
    def from_resources(resources: List[Resource]) -> dict:
        resource_dict = defaultdict(list)
        for resource in resources:
            for resource_type, values in resource.to_dict().items():
                resource_dict[resource_type].extend(values)
        return dict(resource_dict)
