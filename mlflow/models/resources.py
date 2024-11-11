import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import yaml

DEFAULT_API_VERSION = "1"


class ResourceType(Enum):
    """
    Enum to define the different types of resources needed to serve a model.
    """

    UC_CONNECTION = "uc_connection"
    VECTOR_SEARCH_INDEX = "vector_search_index"
    SERVING_ENDPOINT = "serving_endpoint"
    SQL_WAREHOUSE = "sql_warehouse"
    FUNCTION = "function"
    GENIE_SPACE = "genie_space"
    TABLE = "table"


class Resource(ABC):
    """
    Base class for defining the resources needed to serve a model.

    Args:
        type (ResourceType): The resource type.
        target_uri (str): The target URI where these resources are hosted.
    """

    @property
    @abstractmethod
    def type(self) -> ResourceType:
        """
        The resource type (must be defined by subclasses).
        """

    @property
    @abstractmethod
    def target_uri(self) -> str:
        """
        The target URI where the resource is hosted (must be defined by subclasses).
        """

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

    def __eq__(self, other: Any):
        if not isinstance(other, Resource):
            return False
        return self.to_dict() == other.to_dict()


class DatabricksResource(Resource, ABC):
    """
    Base class to define all the Databricks resources to serve a model.

    Example usage: https://docs.databricks.com/en/generative-ai/log-agent.html#specify-resources-for-pyfunc-or-langchain-agent
    """

    @property
    def target_uri(self) -> str:
        return "databricks"


class DatabricksUCConnection(DatabricksResource):
    """
    Define a Databricks UC Connection used to serve a model.

    Args:
        connection_name (str): The name of the databricks UC connection
        used to create the tool which was used to build the model.
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.UC_CONNECTION

    def __init__(self, connection_name: str):
        self.connection_name = connection_name

    def to_dict(self):
        return {self.type.value: [{"name": self.connection_name}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(connection_name=data["name"])


class DatabricksServingEndpoint(DatabricksResource):
    """
    Define Databricks LLM endpoint resource to serve a model.

    Args:
        endpoint_name (str): The name of all the databricks endpoints used by the model.
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.SERVING_ENDPOINT

    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name

    def to_dict(self):
        return {self.type.value: [{"name": self.endpoint_name}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(endpoint_name=data["name"])


class DatabricksVectorSearchIndex(DatabricksResource):
    """
    Define Databricks vector search index name resource to serve a model.

    Args:
        index_name (str): The name of all the databricks vector search index names
        used by the model.
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.VECTOR_SEARCH_INDEX

    def __init__(self, index_name: str):
        self.index_name = index_name

    def to_dict(self):
        return {self.type.value: [{"name": self.index_name}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(index_name=data["name"])


class DatabricksSQLWarehouse(DatabricksResource):
    """
    Define Databricks sql warehouse resource to serve a model.

    Args:
        warehouse_id (str): The id of the sql warehouse used by the model
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.SQL_WAREHOUSE

    def __init__(self, warehouse_id: str):
        self.warehouse_id = warehouse_id

    def to_dict(self):
        return {self.type.value: [{"name": self.warehouse_id}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(warehouse_id=data["name"])


class DatabricksFunction(DatabricksResource):
    """
    Define Databricks UC Function to serve a model.

    Args:
        function_name (str): The name of the function used by the model
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.FUNCTION

    def __init__(self, function_name: str):
        self.function_name = function_name

    def to_dict(self):
        return {self.type.value: [{"name": self.function_name}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(function_name=data["name"])


class DatabricksGenieSpace(DatabricksResource):
    """
    Define a Databricks Genie Space to serve a model.

    Args:
        genie_space_id (str): The genie space id
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.GENIE_SPACE

    def __init__(self, genie_space_id: str):
        self.genie_space_id = genie_space_id

    def to_dict(self):
        return {self.type.value: [{"name": self.genie_space_id}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(genie_space_id=data["name"])


class DatabricksTable(DatabricksResource):
    """
    Defines a Databricks Unity Catalog (UC) Table, which establishes table dependencies
    for Model Serving. This table will be referenced in Agent Model Serving endpoints,
    where an agent queries a SQL table via either Genie or UC Functions.

     Args:
         table_name (str): The name of the table used by the model
    """

    @property
    def type(self) -> ResourceType:
        return ResourceType.TABLE

    def __init__(self, table_name: str):
        self.table_name = table_name

    def to_dict(self):
        return {self.type.value: [{"name": self.table_name}]}

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        return cls(table_name=data["name"])


def _get_resource_class_by_type(target_uri: str, resource_type: ResourceType):
    resource_classes = {
        "databricks": {
            ResourceType.UC_CONNECTION.value: DatabricksUCConnection,
            ResourceType.SERVING_ENDPOINT.value: DatabricksServingEndpoint,
            ResourceType.VECTOR_SEARCH_INDEX.value: DatabricksVectorSearchIndex,
            ResourceType.SQL_WAREHOUSE.value: DatabricksSQLWarehouse,
            ResourceType.FUNCTION.value: DatabricksFunction,
            ResourceType.GENIE_SPACE.value: DatabricksGenieSpace,
            ResourceType.TABLE.value: DatabricksTable,
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
        resources: list[Resource], api_version: str = DEFAULT_API_VERSION
    ) -> dict[str, dict[ResourceType, list[dict]]]:
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
    def from_dict(data) -> dict[str, dict[ResourceType, list[dict]]]:
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
    def from_yaml_file(path: str) -> dict[str, dict[ResourceType, list[dict]]]:
        if not os.path.exists(path):
            raise OSError(f"No such file or directory: '{path}'")
        path = os.path.abspath(path)
        with open(path) as file:
            data = yaml.safe_load(file)
            return _ResourceBuilder.from_dict(data)
