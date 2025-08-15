"""
Registry functions for managing Judge versions similar to Prompt Registry.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from mlflow.genai.judges.base import Judge
from mlflow.utils.annotations import experimental


# Simple in-memory registry for Phase 1
# Future: This will be backed by MLflow Model Registry or tracking store
_JUDGE_REGISTRY: Dict[str, Dict[int, Judge]] = {}
_JUDGE_ALIASES: Dict[str, Dict[str, int]] = {}
_JUDGE_VERSION_COUNTER: Dict[str, int] = {}


@dataclass
class JudgeVersion:
    """
    Represents a specific version of a Judge.
    """
    name: str
    version: int
    judge: Judge
    commit_message: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@experimental(version="3.4.0")
def register_judge(
    name: str,
    instructions: str,
    model: Optional[str] = None,
    examples: Optional[List[Dict]] = None,
    commit_message: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> JudgeVersion:
    """
    Register a new Judge or create a new version of an existing Judge.
    
    Args:
        name: Name of the judge
        instructions: Human-readable instructions for the judge
        model: LLM model to use (defaults to configured default)
        examples: Optional few-shot examples for the judge
        commit_message: Message describing changes (for versioning)
        tags: Version-specific tags
        
    Returns:
        JudgeVersion object with version number
        
    Example:
        >>> # Create v1
        >>> judge_v1 = register_judge(
        ...     name="formality_judge",
        ...     instructions="Check if response is formal",
        ...     model="openai/gpt-4o-mini"
        ... )
        >>> print(f"Created {judge_v1.name} v{judge_v1.version}")
        >>> 
        >>> # After alignment, create v2 with examples
        >>> judge_v2 = register_judge(
        ...     name="formality_judge",
        ...     instructions="Check if response is formal",
        ...     examples=[
        ...         {
        ...             "inputs": {"question": "How to reset?"},
        ...             "outputs": "Hey! Just click reset.",
        ...             "assessment": False  # Informal
        ...         },
        ...         {
        ...             "inputs": {"question": "How to reset?"},
        ...             "outputs": "Please navigate to Settings and select Reset.",
        ...             "assessment": True  # Formal
        ...         }
        ...     ],
        ...     commit_message="Added aligned examples from human feedback"
        ... )
        >>> print(f"Created {judge_v2.name} v{judge_v2.version}")
    """
    from mlflow.genai.judges.utils import get_default_model
    
    # Use default model if not specified
    if model is None:
        model = get_default_model()
    
    # Initialize registry for this judge name if needed
    if name not in _JUDGE_REGISTRY:
        _JUDGE_REGISTRY[name] = {}
        _JUDGE_ALIASES[name] = {}
        _JUDGE_VERSION_COUNTER[name] = 0
    
    # Increment version counter
    _JUDGE_VERSION_COUNTER[name] += 1
    version = _JUDGE_VERSION_COUNTER[name]
    
    # Create the judge instance
    judge = Judge(
        name=name,
        instructions=instructions,
        model=model,
        version=version,
        examples=examples,
    )
    
    # Store in registry
    _JUDGE_REGISTRY[name][version] = judge
    
    # Return version object
    return JudgeVersion(
        name=name,
        version=version,
        judge=judge,
        commit_message=commit_message,
        tags=tags,
    )


@experimental(version="3.4.0")
def load_judge(
    name_or_uri: str,
    version: Optional[Union[str, int]] = None,
) -> Judge:
    """
    Load a Judge from the registry.
    
    Args:
        name_or_uri: Name or URI like "judges:/name/version" or "judges:/name@alias"
        version: Version number or alias (when using name)
        
    Returns:
        Judge instance
        
    Example:
        >>> # Load specific version by name and version
        >>> judge = load_judge("formality_judge", version=2)
        >>> 
        >>> # Load by URI with version
        >>> judge = load_judge("judges:/formality_judge/2")
        >>> 
        >>> # Load by alias
        >>> judge = load_judge("judges:/formality_judge@production")
        >>> 
        >>> # Load latest version (if no version specified)
        >>> judge = load_judge("formality_judge")
    """
    # Parse URI format
    if name_or_uri.startswith("judges:/"):
        uri_parts = name_or_uri[8:]  # Remove "judges:/" prefix
        
        if "@" in uri_parts:
            # Format: judges:/name@alias
            name, alias = uri_parts.split("@", 1)
            if name not in _JUDGE_ALIASES or alias not in _JUDGE_ALIASES[name]:
                raise ValueError(f"Alias '{alias}' not found for judge '{name}'")
            version = _JUDGE_ALIASES[name][alias]
        elif "/" in uri_parts:
            # Format: judges:/name/version
            name, version_str = uri_parts.split("/", 1)
            version = int(version_str)
        else:
            # Format: judges:/name (load latest)
            name = uri_parts
            version = None
    else:
        # Plain name format
        name = name_or_uri
        
        # If version is a string, check if it's an alias
        if isinstance(version, str):
            if name not in _JUDGE_ALIASES or version not in _JUDGE_ALIASES[name]:
                raise ValueError(f"Alias '{version}' not found for judge '{name}'")
            version = _JUDGE_ALIASES[name][version]
    
    # Check if judge exists
    if name not in _JUDGE_REGISTRY:
        raise ValueError(f"Judge '{name}' not found in registry")
    
    # If no version specified, load latest
    if version is None:
        version = max(_JUDGE_REGISTRY[name].keys())
    
    # Check if version exists
    if version not in _JUDGE_REGISTRY[name]:
        raise ValueError(f"Version {version} not found for judge '{name}'")
    
    return _JUDGE_REGISTRY[name][version]


@experimental(version="3.4.0")
def set_judge_alias(name: str, alias: str, version: int) -> None:
    """
    Set an alias (e.g., 'production') for a judge version.
    
    Args:
        name: Name of the judge
        alias: Alias to set (e.g., 'production', 'staging', 'latest')
        version: Version number to associate with the alias
        
    Example:
        >>> # Set production alias
        >>> set_judge_alias("formality_judge", "production", version=2)
        >>> 
        >>> # Now can load using alias
        >>> prod_judge = load_judge("judges:/formality_judge@production")
        >>> 
        >>> # Update alias to point to newer version
        >>> set_judge_alias("formality_judge", "production", version=3)
    """
    if name not in _JUDGE_REGISTRY:
        raise ValueError(f"Judge '{name}' not found in registry")
    
    if version not in _JUDGE_REGISTRY[name]:
        raise ValueError(f"Version {version} not found for judge '{name}'")
    
    if name not in _JUDGE_ALIASES:
        _JUDGE_ALIASES[name] = {}
    
    _JUDGE_ALIASES[name][alias] = version


@experimental(version="3.4.0")
def delete_judge_alias(name: str, alias: str) -> None:
    """
    Delete an alias for a judge.
    
    Args:
        name: Name of the judge
        alias: Alias to delete
        
    Example:
        >>> # Remove staging alias
        >>> delete_judge_alias("formality_judge", "staging")
    """
    if name not in _JUDGE_ALIASES:
        raise ValueError(f"No aliases found for judge '{name}'")
    
    if alias not in _JUDGE_ALIASES[name]:
        raise ValueError(f"Alias '{alias}' not found for judge '{name}'")
    
    del _JUDGE_ALIASES[name][alias]


@experimental(version="3.4.0")
def list_judges() -> List[str]:
    """
    List all registered judge names.
    
    Returns:
        List of judge names
        
    Example:
        >>> judges = list_judges()
        >>> print(f"Registered judges: {judges}")
        ['formality_judge', 'accuracy_judge', 'relevance_judge']
    """
    return list(_JUDGE_REGISTRY.keys())


@experimental(version="3.4.0")
def list_judge_versions(name: str) -> List[int]:
    """
    List all versions of a specific judge.
    
    Args:
        name: Name of the judge
        
    Returns:
        List of version numbers
        
    Example:
        >>> versions = list_judge_versions("formality_judge")
        >>> print(f"Available versions: {versions}")
        [1, 2, 3]
    """
    if name not in _JUDGE_REGISTRY:
        raise ValueError(f"Judge '{name}' not found in registry")
    
    return sorted(_JUDGE_REGISTRY[name].keys())


@experimental(version="3.4.0")
def get_judge_aliases(name: str) -> Dict[str, int]:
    """
    Get all aliases for a judge.
    
    Args:
        name: Name of the judge
        
    Returns:
        Dictionary mapping alias names to version numbers
        
    Example:
        >>> aliases = get_judge_aliases("formality_judge")
        >>> print(aliases)
        {'production': 3, 'staging': 2, 'latest': 3}
    """
    if name not in _JUDGE_ALIASES:
        return {}
    
    return dict(_JUDGE_ALIASES[name])