import logging

from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
from databricks_cli.configure import provider


_logger = logging.getLogger(__name__)


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


def _fail_malformed_databricks_auth(profile):
    raise MlflowException("Got malformed Databricks CLI profile '%s'. Please make sure the "
                          "Databricks CLI is properly configured as described at "
                          "https://github.com/databricks/databricks-cli." % profile)


def get_databricks_host_creds(profile=None):
    """
    Reads in configuration necessary to make HTTP requests to a Databricks server. This
    uses the Databricks CLI's ConfigProvider interface to load the DatabricksConfig object.
    This method will throw an exception if sufficient auth cannot be found.

    :param profile: Databricks CLI profile. If not provided, we will read the default profile.
    :return: :py:class:`mlflow.rest_utils.MlflowHostCreds` which includes the hostname and
        authentication information necessary to talk to the Databricks server.
    """
    if not hasattr(provider, 'get_config'):
        _logger.warning(
            "Support for databricks-cli<0.8.0 is deprecated and will be removed"
            " in a future version.")
        config = provider.get_config_for_profile(profile)
    elif profile:
        config = provider.ProfileConfigProvider(profile).get_config()
    else:
        config = provider.get_config()
    if not config or not config.host:
        _fail_malformed_databricks_auth(profile)

    insecure = hasattr(config, 'insecure') and config.insecure

    if config.username is not None and config.password is not None:
        return MlflowHostCreds(config.host, username=config.username, password=config.password,
                               ignore_tls_verification=insecure)
    elif config.token:
        return MlflowHostCreds(config.host, token=config.token, ignore_tls_verification=insecure)
    _fail_malformed_databricks_auth(profile)
