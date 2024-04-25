from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ResourceType(Enum):
    """
    Enum to define the different types of resources needed to serve a model.
    """

    VECTOR_SEARCH_INDEX = "vector_search_index"
    SERVING_ENDPOINT = "serving_endpoint"


@dataclass
class Resource(ABC):
    """
    Base class for defining the resources needed to serve a model.

    Args:
        type (ResourceType): The resource type.
        target_uri (str): The target URI where these resources are hosted.
    """

    type: ResourceType
    target_uri: str

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

    target_uri: str = "databricks"


@dataclass
class DatabricksServingEndpoint(DatabricksResource):
    """
    Define Databricks LLM endpoint resource to serve a model.

    Args:
        endpoint_name (str): The name of all the databricks endpoints used by the model.
    """

    type: ResourceType = ResourceType.SERVING_ENDPOINT
    endpoint_name: str = None

    def to_dict(self):
        return {self.type.value: [{"name": self.endpoint_name}]} if self.endpoint_name else {}


@dataclass
class DatabricksVectorSearchIndex(DatabricksResource):
    """
    Define Databricks vector search index name resource to serve a model.

    Args:
        index_name (str): The name of all the databricks vector search index names
        used by the model.
    """

    type: ResourceType = ResourceType.VECTOR_SEARCH_INDEX
    index_name: str = None

    def to_dict(self):
        return {self.type.value: [{"name": self.index_name}]} if self.index_name else {}


class _ResourceBuilder:
    """
    Private builder class to build the resources dictionary.
    """

    @staticmethod
    def from_resources(resources: List[Resource]) -> Dict[str, Dict[ResourceType, List[str]]]:
        resource_dict = defaultdict(lambda: defaultdict(list))
        for resource in resources:
            resource_data = resource.to_dict()
            for resource_type, values in resource_data.items():
                resource_dict[resource.target_uri][resource_type].extend(values)
        return dict(resource_dict)
