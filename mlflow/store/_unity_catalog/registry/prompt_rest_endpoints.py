"""
Endpoint utilities for Unity Catalog Prompt Registry REST API.
"""

from typing import Tuple
from mlflow.protos.unity_catalog_prompt_service_pb2 import UnityCatalogPromptService
from mlflow.utils.rest_utils import (
    _UC_REST_API_PATH_PREFIX,
    extract_api_info_for_service,
    extract_all_api_info_for_service,
)

# Base path for UC prompt endpoints
_UC_PROMPT_API_BASE_PATH = "/mlflow/unity-catalog/prompts"

# Extract endpoint info from proto service
_METHOD_TO_INFO = extract_api_info_for_service(UnityCatalogPromptService, _UC_PROMPT_API_BASE_PATH)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(UnityCatalogPromptService, _UC_PROMPT_API_BASE_PATH)

def resolve_endpoint(service: str, **kwargs) -> Tuple[str, str]:
    """
    Resolve the endpoint path and HTTP method for a service.
    
    Args:
        service: The service name (e.g. "CreatePrompt")
        **kwargs: Path parameters to substitute into the endpoint path
        
    Returns:
        Tuple of (endpoint_path, http_method)
    """
    endpoint, method = _METHOD_TO_INFO[service]
    
    # Substitute path parameters
    if kwargs:
        endpoint = endpoint.format(**kwargs)
        
    return endpoint, method 