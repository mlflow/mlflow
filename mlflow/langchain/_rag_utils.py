"""
This module contains the utility functions and constants that are specifically
used by the databricks RAG to log chain as code along with the config.
"""

from typing import Optional

# The following constant is used in flavor configuration to specify the code paths
# that should be set using _set_config_path when the model is loaded by code.
_CODE_CONFIG = "_code_config"
_CODE_PATH = "_code_path"

__databricks_rag_chain__ = None
__databricks_rag_config_path__ = None


def _set_chain(chain):
    """
    The function is used to set the chain by the users of
    the library at a global level. This global state then
    can be referenced by the library to get the chain instance.
    """
    globals()["__databricks_rag_chain__"] = chain


def _set_config_path(path: Optional[str] = None):
    """
    The function is used by the library to provide the local
    path of the code path to the users so it can be referenced
    while loading the chain back.
    """
    globals()["__databricks_rag_config_path__"] = path
