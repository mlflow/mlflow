"""
Unity Catalog Prompt Registry Store implementation using direct REST endpoints.
"""

import functools
import logging
from typing import List, Optional, Dict

from mlflow.entities.model_registry import Prompt, PromptVersion
from mlflow.environment_variables import MLFLOW_ENABLE_UC_PROMPT_SUPPORT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store._unity_catalog.registry.rest_store import BaseRestStore
from mlflow.store._unity_catalog.registry.prompt_rest_endpoints import resolve_endpoint
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils._spark_utils import _get_active_spark_session

_logger = logging.getLogger(__name__)

class UcPromptRestStore(BaseRestStore):
    """
    REST store for Unity Catalog Prompts using direct prompt-specific endpoints.
    """
    
    def __init__(self, store_uri):
        """
        Initialize the UC prompt REST store.
        
        Args:
            store_uri: URI with scheme 'databricks-uc'
        """
        if not MLFLOW_ENABLE_UC_PROMPT_SUPPORT.get():
            raise MlflowException(
                "Unity Catalog prompt support is not enabled. "
                "Set MLFLOW_ENABLE_UC_PROMPT_SUPPORT=true to enable it.",
                INVALID_PARAMETER_VALUE,
            )
        
        super().__init__(get_host_creds=functools.partial(get_databricks_host_creds, store_uri))
        self.store_uri = store_uri
        try:
            self.spark = _get_active_spark_session()
        except Exception:
            self.spark = None

    def _call_endpoint(self, service, json_body=None, **kwargs):
        """
        Call REST endpoint with the specified service and parameters.
        
        Args:
            service: Service name (e.g. "CreatePrompt")
            json_body: JSON request body
            **kwargs: Path parameters for endpoint URL
            
        Returns:
            Response proto object
        """
        endpoint, method = resolve_endpoint(service, **kwargs)
        response_proto = self._call_endpoint(endpoint, method, json_body)
        return response_proto

    def create_prompt(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Prompt:
        """
        Create a new prompt in Unity Catalog.
        
        Args:
            name: The name of the prompt
            template: The template text of the prompt
            description: Optional description of the prompt
            tags: Optional dictionary of prompt tags
            
        Returns:
            The created Prompt object
        """
        # Convert tags to proto format
        proto_tags = []
        if tags:
            proto_tags = [{"key": k, "value": v} for k, v in tags.items()]

        # Create request body
        req_body = message_to_json({
            "name": name,
            "prompt": {
                "name": name,
                "description": description,
                "tags": proto_tags,
            }
        })

        response_proto = self._call_endpoint("CreatePrompt", req_body)
        prompt = response_proto.prompt
        
        # Create initial version with template
        version_req_body = message_to_json({
            "name": name,
            "version": "1",
            "prompt_version": {
                "name": name,
                "template": template,
                "description": description,
            }
        })
        
        version_response = self._call_endpoint(
            "CreatePromptVersion", 
            version_req_body,
            name=name
        )
        return Prompt.from_proto(prompt, version_response.prompt_version)

    def get_prompt(self, name: str) -> Prompt:
        """
        Get a prompt by name.
        
        Args:
            name: The name of the prompt
            
        Returns:
            The requested Prompt object
        """
        response_proto = self._call_endpoint("GetPrompt", name=name)
        return Prompt.from_proto(response_proto.prompt)

    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt.
        
        Args:
            name: The name of the prompt to delete
        """
        self._call_endpoint("DeletePrompt", name=name)

    def search_prompts(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
        page_token: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts.
        
        Args:
            filter_string: Filter string like "name='my-prompt'"
            max_results: Maximum number of prompts to return
            page_token: Token for pagination
            experiment_id: Optional experiment ID to filter by
            
        Returns:
            PagedList of Prompt objects
        """
        req_body = message_to_json({
            "filter": filter_string,
            "max_results": max_results,
            "page_token": page_token,
            "experiment_id": experiment_id,
        })

        response_proto = self._call_endpoint("SearchPrompts", req_body)
        prompts = [Prompt.from_proto(p) for p in response_proto.prompts]
        return PagedList(prompts, response_proto.next_page_token)

    def create_prompt_version(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> PromptVersion:
        """
        Create a new version of a prompt.
        
        Args:
            name: The name of the prompt
            template: The template text for this version
            description: Optional description of this version
            tags: Optional dictionary of version tags
            
        Returns:
            The created PromptVersion object
        """
        # Convert tags to proto format
        proto_tags = []
        if tags:
            proto_tags = [{"key": k, "value": v} for k, v in tags.items()]

        req_body = message_to_json({
            "name": name,
            "prompt_version": {
                "name": name,
                "template": template,
                "description": description,
                "tags": proto_tags,
            }
        })

        response_proto = self._call_endpoint(
            "CreatePromptVersion", 
            req_body,
            name=name
        )
        return PromptVersion.from_proto(response_proto.prompt_version)

    def get_prompt_version(self, name: str, version: str) -> PromptVersion:
        """
        Get a specific version of a prompt.
        
        Args:
            name: The name of the prompt
            version: The version number
            
        Returns:
            The requested PromptVersion object
        """
        response_proto = self._call_endpoint(
            "GetPromptVersion",
            name=name,
            version=version
        )
        return PromptVersion.from_proto(response_proto.prompt_version)

    def delete_prompt_version(self, name: str, version: str) -> None:
        """
        Delete a specific version of a prompt.
        
        Args:
            name: The name of the prompt
            version: The version number to delete
        """
        self._call_endpoint(
            "DeletePromptVersion",
            name=name,
            version=version
        )

    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt.
        
        Args:
            name: The name of the prompt
            key: Tag key
            value: Tag value
        """
        req_body = message_to_json({
            "key": key,
            "value": value,
        })
        self._call_endpoint(
            "SetPromptTag",
            req_body,
            name=name
        )

    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt.
        
        Args:
            name: The name of the prompt
            key: The tag key to delete
        """
        self._call_endpoint(
            "DeletePromptTag",
            name=name,
            key=key
        )

    def set_prompt_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt version.
        
        Args:
            name: The name of the prompt
            version: The version number
            key: Tag key
            value: Tag value
        """
        req_body = message_to_json({
            "key": key,
            "value": value,
        })
        self._call_endpoint(
            "SetPromptVersionTag",
            req_body,
            name=name,
            version=version
        )

    def delete_prompt_version_tag(self, name: str, version: str, key: str) -> None:
        """
        Delete a tag from a prompt version.
        
        Args:
            name: The name of the prompt
            version: The version number
            key: The tag key to delete
        """
        self._call_endpoint(
            "DeletePromptVersionTag",
            name=name,
            version=version,
            key=key
        )

    def set_prompt_alias(self, name: str, alias: str, version: str) -> None:
        """
        Set an alias for a prompt version.
        
        Args:
            name: The name of the prompt
            alias: The alias to set
            version: The version to alias
        """
        req_body = message_to_json({
            "version": version,
        })
        self._call_endpoint(
            "SetPromptAlias",
            req_body,
            name=name,
            alias=alias
        )

    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete a prompt alias.
        
        Args:
            name: The name of the prompt
            alias: The alias to delete
        """
        self._call_endpoint(
            "DeletePromptAlias",
            name=name,
            alias=alias
        )

    def get_prompt_version_by_alias(self, name: str, alias: str) -> PromptVersion:
        """
        Get a prompt version by alias.
        
        Args:
            name: The name of the prompt
            alias: The alias to look up
            
        Returns:
            The PromptVersion object referenced by the alias
        """
        response_proto = self._call_endpoint(
            "GetPromptVersionByAlias",
            name=name,
            alias=alias
        )
        return PromptVersion.from_proto(response_proto.prompt_version) 