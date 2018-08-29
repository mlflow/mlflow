def _get_dbutils():
    try:
        import IPython
        ip_shell = IPython.get_ipython()
        if ip_shell is None:
            raise _NoDbutilsError
        return ip_shell.ns_table["user_global"]["dbutils"]
    except ImportError:
        raise _NoDbutilsError
    except KeyError:
        raise _NoDbutilsError


class _NoDbutilsError(Exception):
    pass


def _get_extra_context(context_key):
    dbutils = _get_dbutils()
    java_dbutils = dbutils.notebook.entry_point.getDbutils()
    return java_dbutils.notebook().getContext().extraContext().get(context_key).get()


def is_in_databricks_notebook():
    try:
        return _get_extra_context("aclPathOfAclRoot").startswith('/workspace')
    except Exception:  # pylint: disable=broad-except
        return False


def get_notebook_id():
    """Should only be called if is_in_databricks_notebook is true"""
    acl_path = _get_extra_context("aclPathOfAclRoot")
    if acl_path.startswith('/workspace'):
        return acl_path.split('/')[-1]
    return None


def get_notebook_path():
    """Should only be called if is_in_databricks_notebook is true"""
    return _get_extra_context("notebook_path")


def get_webapp_url():
    """Should only be called if is_in_databricks_notebook is true"""
    return _get_extra_context("api_url")
