import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import yaml

DEFAULT_API_VERSION = "1"


class ResourceType(Enum):
    """
    Enum to define the different types of resources needed to serve a model.
    """

    VECTOR_SEARCH_INDEX = "vector_search_index"
    SERVING_ENDPOINT = "serving_endpoint"
    SQL_WAREHOUSE = "sql_warehouse"
    FUNCTION = "function"


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

    @classmethod
    def from_dict(cls, data):
        """
        Convert the dictionary to a Resource.
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

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(endpoint_name=data["name"])


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

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(index_name=data["name"])


@dataclass
class DatabricksSQLWarehouse(DatabricksResource):
    """
    Define Databricks sql warehouse resource to serve a model.

    Args:
        warehouse_id (str): The id of the sql warehouse used by the model
    """

    type: ResourceType = ResourceType.SQL_WAREHOUSE
    warehouse_id: str = None

    def to_dict(self):
        return {self.type.value: [{"name": self.warehouse_id}]} if self.warehouse_id else {}

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(warehouse_id=data["name"])


@dataclass
class DatabricksFunction(DatabricksResource):
    """
    Define Databricks UC Function to serve a model.

    Args:
        function_name (str): The name of the function used by the model
    """

    type: ResourceType = ResourceType.FUNCTION
    function_name: str = None

    def to_dict(self):
        return {self.type.value: [{"name": self.function_name}]} if self.function_name else {}

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(function_name=data["name"])


def _get_resource_class_by_type(target_uri: str, resource_type: ResourceType):
    resource_classes = {
        "databricks": {
            ResourceType.SERVING_ENDPOINT.value: DatabricksServingEndpoint,
            ResourceType.VECTOR_SEARCH_INDEX.value: DatabricksVectorSearchIndex,
            ResourceType.SQL_WAREHOUSE.value: DatabricksSQLWarehouse,
            ResourceType.FUNCTION.value: DatabricksFunction,
        }
    }
    resource = resource_classes.get(target_uri)
    if resource is None:
        raise ValueError(f"Unsupported target URI: {target_uri}")
    return resource.get(resource_type)


class _ResourceBuilder:
    """
    Private builder class to build the resources dictionary.
    """

    @staticmethod
    def from_resources(
        resources: List[Resource], api_version: str = DEFAULT_API_VERSION
    ) -> Dict[str, Dict[ResourceType, List[Dict]]]:
        resource_dict = {}
        for resource in resources:
            resource_data = resource.to_dict()
            for resource_type, values in resource_data.items():
                target_dict = resource_dict.setdefault(resource.target_uri, {})
                target_list = target_dict.setdefault(resource_type, [])
                target_list.extend(values)

        resource_dict["api_version"] = api_version
        return resource_dict

    @staticmethod
    def from_dict(data) -> Dict[str, Dict[ResourceType, List[Dict]]]:
        resources = []
        api_version = data.pop("api_version")
        if api_version == "1":
            for target_uri, config in data.items():
                for resource_type, values in config.items():
                    resource_class = _get_resource_class_by_type(target_uri, resource_type)
                    if resource_class:
                        resources.extend(resource_class.from_dict(value) for value in values)
                    else:
                        raise ValueError(f"Unsupported resource type: {resource_type}")
        else:
            raise ValueError(f"Unsupported API version: {api_version}")

        return _ResourceBuilder.from_resources(resources, api_version)

    @staticmethod
    def from_yaml_file(path: str) -> Dict[str, Dict[ResourceType, List[Dict]]]:
        if not os.path.exists(path):
            raise OSError(f"No such file or directory: '{path}'")
        path = os.path.abspath(path)
        with open(path) as file:
            data = yaml.safe_load(file)
            return _ResourceBuilder.from_dict(data)
