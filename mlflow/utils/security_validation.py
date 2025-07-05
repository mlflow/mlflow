"""
Genesis-Flow Security Validation

This module provides comprehensive input validation to prevent
injection attacks, path traversal, and other security vulnerabilities.
"""

import os
import re
import logging
from pathlib import Path
from typing import Union, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class SecurityValidationError(Exception):
    """Exception raised for security validation failures."""
    pass

class InputValidator:
    """
    Comprehensive input validation for Genesis-Flow.
    """
    
    # Safe characters for different input types
    SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    SAFE_TAG_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.\/]+$')
    SAFE_TAG_VALUE_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.\s\/\:\@]+$')
    
    # Maximum lengths to prevent DoS
    MAX_NAME_LENGTH = 256
    MAX_TAG_KEY_LENGTH = 250
    MAX_TAG_VALUE_LENGTH = 5000
    MAX_DESCRIPTION_LENGTH = 10000
    MAX_ARTIFACT_PATH_LENGTH = 1000
    
    @staticmethod
    def validate_experiment_name(name: str) -> str:
        """
        Validate experiment name against injection attacks.
        
        Args:
            name: Experiment name to validate
            
        Returns:
            Validated name
            
        Raises:
            SecurityValidationError: If name is invalid
        """
        if not name or not isinstance(name, str):
            raise SecurityValidationError("Experiment name must be a non-empty string")
        
        if len(name) > InputValidator.MAX_NAME_LENGTH:
            raise SecurityValidationError(
                f"Experiment name too long (max {InputValidator.MAX_NAME_LENGTH} chars)"
            )
        
        # Check for dangerous characters
        if not InputValidator.SAFE_NAME_PATTERN.match(name):
            raise SecurityValidationError(
                "Experiment name contains invalid characters. "
                "Only alphanumeric, dash, underscore, and dot allowed."
            )
        
        # Prevent directory traversal
        if '..' in name or '/' in name or '\\' in name:
            raise SecurityValidationError("Experiment name cannot contain path traversal sequences")
        
        logger.debug(f"Validated experiment name: {name}")
        return name
    
    @staticmethod
    def validate_run_name(name: Optional[str]) -> Optional[str]:
        """
        Validate run name.
        
        Args:
            name: Run name to validate (can be None)
            
        Returns:
            Validated name or None
        """
        if name is None:
            return None
            
        if not isinstance(name, str):
            raise SecurityValidationError("Run name must be a string")
        
        if len(name) > InputValidator.MAX_NAME_LENGTH:
            raise SecurityValidationError(
                f"Run name too long (max {InputValidator.MAX_NAME_LENGTH} chars)"
            )
        
        # Allow more flexible run names but still prevent injection
        if '..' in name or '\x00' in name:
            raise SecurityValidationError("Run name contains invalid sequences")
        
        return name
    
    @staticmethod
    def validate_tag_key(key: str) -> str:
        """
        Validate tag key.
        
        Args:
            key: Tag key to validate
            
        Returns:
            Validated key
        """
        if not key or not isinstance(key, str):
            raise SecurityValidationError("Tag key must be a non-empty string")
        
        if len(key) > InputValidator.MAX_TAG_KEY_LENGTH:
            raise SecurityValidationError(
                f"Tag key too long (max {InputValidator.MAX_TAG_KEY_LENGTH} chars)"
            )
        
        if not InputValidator.SAFE_TAG_KEY_PATTERN.match(key):
            raise SecurityValidationError(
                "Tag key contains invalid characters"
            )
        
        return key
    
    @staticmethod
    def validate_tag_value(value: str) -> str:
        """
        Validate tag value.
        
        Args:
            value: Tag value to validate
            
        Returns:
            Validated value
        """
        if not isinstance(value, str):
            raise SecurityValidationError("Tag value must be a string")
        
        if len(value) > InputValidator.MAX_TAG_VALUE_LENGTH:
            raise SecurityValidationError(
                f"Tag value too long (max {InputValidator.MAX_TAG_VALUE_LENGTH} chars)"
            )
        
        # Check for control characters and potential injection
        if '\x00' in value or '\r' in value:
            raise SecurityValidationError("Tag value contains invalid control characters")
        
        return value
    
    @staticmethod
    def validate_artifact_path(path: str, base_path: Optional[str] = None) -> str:
        """
        Validate artifact path to prevent path traversal attacks.
        
        Args:
            path: Artifact path to validate
            base_path: Optional base path for additional validation
            
        Returns:
            Validated path
            
        Raises:
            SecurityValidationError: If path is unsafe
        """
        if not path or not isinstance(path, str):
            raise SecurityValidationError("Artifact path must be a non-empty string")
        
        if len(path) > InputValidator.MAX_ARTIFACT_PATH_LENGTH:
            raise SecurityValidationError(
                f"Artifact path too long (max {InputValidator.MAX_ARTIFACT_PATH_LENGTH} chars)"
            )
        
        # Normalize the path to resolve any relative components
        normalized_path = os.path.normpath(path)
        
        # Check for path traversal attempts
        if '..' in normalized_path or normalized_path.startswith('/'):
            raise SecurityValidationError("Artifact path contains path traversal sequences")
        
        # Additional check if base path provided
        if base_path:
            base = Path(base_path).resolve()
            try:
                target = (base / normalized_path).resolve()
                # Ensure target is within base directory
                target.relative_to(base)
            except ValueError:
                raise SecurityValidationError("Artifact path escapes base directory")
        
        logger.debug(f"Validated artifact path: {normalized_path}")
        return normalized_path
    
    @staticmethod
    def validate_metric_key(key: str) -> str:
        """
        Validate metric key.
        
        Args:
            key: Metric key to validate
            
        Returns:
            Validated key
        """
        if not key or not isinstance(key, str):
            raise SecurityValidationError("Metric key must be a non-empty string")
        
        if len(key) > InputValidator.MAX_TAG_KEY_LENGTH:
            raise SecurityValidationError(
                f"Metric key too long (max {InputValidator.MAX_TAG_KEY_LENGTH} chars)"
            )
        
        # Metric keys should be more restrictive
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', key):
            raise SecurityValidationError(
                "Metric key contains invalid characters. "
                "Only alphanumeric, dash, underscore, and dot allowed."
            )
        
        return key
    
    @staticmethod
    def validate_param_key(key: str) -> str:
        """
        Validate parameter key.
        
        Args:
            key: Parameter key to validate
            
        Returns:
            Validated key
        """
        if not key or not isinstance(key, str):
            raise SecurityValidationError("Parameter key must be a non-empty string")
        
        if len(key) > InputValidator.MAX_TAG_KEY_LENGTH:
            raise SecurityValidationError(
                f"Parameter key too long (max {InputValidator.MAX_TAG_KEY_LENGTH} chars)"
            )
        
        # Parameter keys should be restrictive
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', key):
            raise SecurityValidationError(
                "Parameter key contains invalid characters. "
                "Only alphanumeric, dash, underscore, and dot allowed."
            )
        
        return key
    
    @staticmethod
    def validate_param_value(value: Union[str, int, float]) -> str:
        """
        Validate parameter value.
        
        Args:
            value: Parameter value to validate
            
        Returns:
            Validated value as string
        """
        if value is None:
            raise SecurityValidationError("Parameter value cannot be None")
        
        # Convert to string if needed
        str_value = str(value)
        
        if len(str_value) > InputValidator.MAX_TAG_VALUE_LENGTH:
            raise SecurityValidationError(
                f"Parameter value too long (max {InputValidator.MAX_TAG_VALUE_LENGTH} chars)"
            )
        
        # Check for control characters
        if '\x00' in str_value or '\r' in str_value:
            raise SecurityValidationError("Parameter value contains invalid control characters")
        
        return str_value
    
    @staticmethod
    def validate_model_name(name: str) -> str:
        """
        Validate registered model name.
        
        Args:
            name: Model name to validate
            
        Returns:
            Validated name
        """
        if not name or not isinstance(name, str):
            raise SecurityValidationError("Model name must be a non-empty string")
        
        if len(name) > InputValidator.MAX_NAME_LENGTH:
            raise SecurityValidationError(
                f"Model name too long (max {InputValidator.MAX_NAME_LENGTH} chars)"
            )
        
        # Model names should be safe for file systems and URLs
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', name):
            raise SecurityValidationError(
                "Model name contains invalid characters. "
                "Only alphanumeric, dash, underscore, and dot allowed."
            )
        
        # Prevent directory traversal
        if '..' in name:
            raise SecurityValidationError("Model name cannot contain '..'")
        
        return name
    
    @staticmethod
    def validate_uri(uri: str, allowed_schemes: Optional[set] = None) -> str:
        """
        Validate URI for safety.
        
        Args:
            uri: URI to validate
            allowed_schemes: Set of allowed URI schemes
            
        Returns:
            Validated URI
        """
        if not uri or not isinstance(uri, str):
            raise SecurityValidationError("URI must be a non-empty string")
        
        if allowed_schemes is None:
            allowed_schemes = {'http', 'https', 'file', 's3', 'gs', 'azure'}
        
        try:
            parsed = urlparse(uri)
        except Exception as e:
            raise SecurityValidationError(f"Invalid URI format: {e}")
        
        if parsed.scheme and parsed.scheme not in allowed_schemes:
            raise SecurityValidationError(
                f"URI scheme '{parsed.scheme}' not allowed. "
                f"Allowed schemes: {sorted(allowed_schemes)}"
            )
        
        # Check for potential SSRF attacks
        if parsed.hostname:
            # Block private/local addresses for HTTP schemes
            if parsed.scheme in ('http', 'https'):
                if parsed.hostname.lower() in ('localhost', '127.0.0.1', '0.0.0.0'):
                    raise SecurityValidationError("URI points to localhost/loopback address")
                
                # Block private IP ranges (basic check)
                if (parsed.hostname.startswith('192.168.') or 
                    parsed.hostname.startswith('10.') or
                    parsed.hostname.startswith('172.')):
                    raise SecurityValidationError("URI points to private IP address")
        
        return uri

def secure_filename(filename: str) -> str:
    """
    Secure a filename by removing/replacing dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Secured filename
    """
    if not filename:
        raise SecurityValidationError("Filename cannot be empty")
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Ensure it doesn't start with dot (hidden file)
    if filename.startswith('.'):
        filename = '_' + filename[1:]
    
    # Ensure reasonable length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:251-len(ext)] + ext
    
    return filename