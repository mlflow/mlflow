__databricks_rag_chain__ = None
__databricks_rag_config_path__ = None


def _set_chain(chain):
    """
    The function is used to set the chain by the users of
    the library at a global level. This global state then
    can be referenced by the library to get the chain instance.
    """
    globals()["__databricks_rag_chain__"] = chain


def _set_code_path(path):
    """
    The function is used by the library to provide the local
    path of the code path to the users so it can be referenced
    while loading the chain back.
    """
    globals()["__databricks_rag_config_path__"] = path
