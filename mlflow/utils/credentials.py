import configparser
import getpass
import logging
import os
from typing import NamedTuple

from mlflow.environment_variables import (
    MLFLOW_ENTRA_ID_SCOPE,
    MLFLOW_TRACKING_AUTH,
    MLFLOW_TRACKING_AWS_SIGV4,
    MLFLOW_TRACKING_CLIENT_CERT_PATH,
    MLFLOW_TRACKING_INSECURE_TLS,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_SERVER_CERT_PATH,
    MLFLOW_TRACKING_TOKEN,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds

_logger = logging.getLogger(__name__)


class MlflowCreds(NamedTuple):
    username: str | None
    password: str | None


def _get_credentials_path() -> str:
    return os.path.expanduser("~/.mlflow/credentials")


def _read_mlflow_creds_from_file() -> tuple[str | None, str | None]:
    path = _get_credentials_path()
    if not os.path.exists(path):
        return None, None

    config = configparser.ConfigParser()
    config.read(path)
    if "mlflow" not in config:
        return None, None

    mlflow_cfg = config["mlflow"]
    username_key = MLFLOW_TRACKING_USERNAME.name.lower()
    password_key = MLFLOW_TRACKING_PASSWORD.name.lower()
    return mlflow_cfg.get(username_key), mlflow_cfg.get(password_key)


def _read_mlflow_creds_from_env() -> tuple[str | None, str | None]:
    return MLFLOW_TRACKING_USERNAME.get(), MLFLOW_TRACKING_PASSWORD.get()


def read_mlflow_creds() -> MlflowCreds:
    username_file, password_file = _read_mlflow_creds_from_file()
    username_env, password_env = _read_mlflow_creds_from_env()
    return MlflowCreds(
        username=username_env or username_file,
        password=password_env or password_file,
    )


def get_default_host_creds(store_uri):
    creds = read_mlflow_creds()
    return MlflowHostCreds(
        host=store_uri,
        username=creds.username,
        password=creds.password,
        token=MLFLOW_TRACKING_TOKEN.get(),
        aws_sigv4=MLFLOW_TRACKING_AWS_SIGV4.get(),
        auth=MLFLOW_TRACKING_AUTH.get(),
        ignore_tls_verification=MLFLOW_TRACKING_INSECURE_TLS.get(),
        client_cert_path=MLFLOW_TRACKING_CLIENT_CERT_PATH.get(),
        server_cert_path=MLFLOW_TRACKING_SERVER_CERT_PATH.get(),
    )


def login(backend: str = "databricks", interactive: bool = True) -> None:
    """Configure MLflow server authentication and connect MLflow to tracking server.

    This method provides a simple way to connect MLflow to its tracking server. The following
    backends are supported:

    - "databricks" (default): Connect to a Databricks tracking server. Users will be prompted
      to enter the credentials if no existing Databricks profile is found, and the credentials
      will be saved to `~/.databrickscfg`.
    - "entra": Connect to an MLflow tracking server protected by Microsoft Entra ID (formerly
      Azure Active Directory). Access tokens are acquired with
      ``azure.identity.DefaultAzureCredential`` (which supports environment credentials,
      managed identities, the Azure CLI login and more) for the scope configured through the
      ``MLFLOW_ENTRA_ID_SCOPE`` environment variable, and attached to every outgoing request.
      Requires the ``azure-identity`` package. Users will be prompted for the token scope and
      the tracking server URI if they are not set via the ``MLFLOW_ENTRA_ID_SCOPE`` and
      ``MLFLOW_TRACKING_URI`` environment variables.

    Args:
        backend: string, the backend of the tracking server. "databricks" and "entra" are
            supported.

        interactive: bool, controls request for user input on missing credentials. If true, user
            input will be requested if no credentials are found, otherwise an exception will be
            raised if no credentials are found.

    .. code-block:: python
        :caption: Example

        import mlflow

        mlflow.login()
        with mlflow.start_run():
            mlflow.log_param("p", 0)
    """
    from mlflow.tracking import set_tracking_uri

    if backend == "databricks":
        _databricks_login(interactive)
        set_tracking_uri("databricks")
    elif backend == "entra":
        host = _entra_login(interactive)
        set_tracking_uri(host)
    else:
        raise MlflowException(
            "Currently only 'databricks' and 'entra' backends are supported, "
            f"received `backend={backend}`."
        )


def _validate_databricks_auth():
    # Check if databricks credentials are valid.
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "Databricks SDK is not installed. To use `mlflow.login()`, please install "
            "databricks-sdk by `pip install databricks-sdk`."
        )

    try:
        w = WorkspaceClient()
        if "community" in w.config.host:
            # Databricks community edition cannot use `w.current_user.me()` for auth validation.
            w.clusters.list_zones()
        else:
            # If credentials are invalid, `w.current_user.me()` will throw an error.
            w.current_user.me()
        _logger.info(
            f"Successfully connected to MLflow hosted tracking server! Host: {w.config.host}."
        )
    except Exception as e:
        raise MlflowException(f"Failed to validate databricks credentials: {e}")


def _overwrite_or_create_databricks_profile(
    file_name,
    profile,
    profile_name="DEFAULT",
):
    """Overwrite or create a profile in the databricks config file.

    Args:
        file_name: string, the file name of the databricks config file, usually `~/.databrickscfg`.
        profile: dict, contains the authentiacation profile information.
        profile_name: string, the name of the profile to be overwritten or created.
    """
    profile_name = f"[{profile_name}]"
    lines = []
    # Read `file_name` if the file exists, otherwise `lines=[]`.
    if os.path.exists(file_name):
        with open(file_name) as file:
            lines = file.readlines()
    start_index = -1
    end_index = -1
    # Find the start and end indices of the profile to overwrite.
    for i in range(len(lines)):
        if lines[i].strip() == profile_name:
            start_index = i
            break

    if start_index != -1:
        for i in range(start_index + 1, len(lines)):
            # Reach an empty line or a new profile.
            if lines[i].strip() == "" or lines[i].startswith("["):
                end_index = i
                break
        end_index = end_index if end_index != -1 else len(lines)
        del lines[start_index : end_index + 1]

    # Write the new profile to the top of the file.
    new_profile = []
    new_profile.append(profile_name + "\n")
    new_profile.append(f"host = {profile['host']}\n")
    if "token" in profile:
        new_profile.append(f"token = {profile['token']}\n")
    else:
        new_profile.append(f"username = {profile['username']}\n")
        new_profile.append(f"password = {profile['password']}\n")
    new_profile.append("\n")
    lines = new_profile + lines

    # Write back the modified lines to the file.
    with open(file_name, "w") as file:
        file.writelines(lines)


def _databricks_login(interactive):
    """Set up databricks authentication."""
    try:
        # Failed validation will throw an error.
        _validate_databricks_auth()
        return
    except Exception:
        if interactive:
            _logger.info("No valid Databricks credentials found, please enter your credentials...")
        else:
            raise MlflowException(
                "No valid Databricks credentials found while running in non-interactive mode."
            )

    while True:
        host = input("Databricks Host (should begin with https://): ")
        if not host.startswith("https://"):
            _logger.error("Invalid host: {host}, host must begin with https://, please retry.")
        break

    profile = {"host": host}
    if "community" in host:
        # Databricks community edition requires username and password for authentication.
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        profile["username"] = username
        profile["password"] = password
    else:
        # Production or staging Databricks requires personal token for authentication.
        token = getpass.getpass("Token: ")
        profile["token"] = token

    file_name = os.environ.get(
        "DATABRICKS_CONFIG_FILE", f"{os.path.expanduser('~')}/.databrickscfg"
    )
    profile_name = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    _overwrite_or_create_databricks_profile(file_name, profile, profile_name)

    try:
        # Failed validation will throw an error.
        _validate_databricks_auth()
    except Exception as e:
        # If user entered invalid auth, we will raise an error and ask users to retry.
        raise MlflowException(f"`mlflow.login()` failed with error: {e}")


def _validate_entra_auth():
    # Check that a Microsoft Entra ID token can be acquired for the configured scope.
    try:
        from azure.identity import DefaultAzureCredential  # noqa: F401
    except ImportError:
        raise ImportError(
            "The azure-identity package is not installed. To use "
            '`mlflow.login(backend="entra")`, please install it by '
            "`pip install azure-identity`."
        )

    from mlflow.tracking.request_auth.entra_request_auth_provider import _get_token

    try:
        # Failed token acquisition will throw an error.
        _get_token()
        _logger.info("Successfully acquired a Microsoft Entra ID token.")
    except Exception as e:
        raise MlflowException(f"Failed to validate Microsoft Entra ID credentials: {e}")


def _entra_login(interactive):
    """Set up Microsoft Entra ID authentication.

    Returns:
        The tracking server URI to connect to.
    """
    if not MLFLOW_ENTRA_ID_SCOPE.get():
        if not interactive:
            raise MlflowException(
                f"{MLFLOW_ENTRA_ID_SCOPE.name} must be set while running in non-interactive mode."
            )
        while True:
            scope = input("Entra ID token scope (e.g. api://<client-id>/.default): ").strip()
            if scope:
                break
            _logger.error("Scope must not be empty, please retry.")
        MLFLOW_ENTRA_ID_SCOPE.set(scope)

    host = MLFLOW_TRACKING_URI.get()
    if not (host and host.startswith(("http://", "https://"))):
        if not interactive:
            raise MlflowException(
                f"{MLFLOW_TRACKING_URI.name} must be set to an HTTP(S) tracking server URI "
                "while running in non-interactive mode."
            )
        while True:
            host = input("MLflow tracking server URI (should begin with http(s)://): ").strip()
            if host.startswith(("http://", "https://")):
                break
            _logger.error(f"Invalid URI: {host}, URI must begin with http(s)://, please retry.")
        MLFLOW_TRACKING_URI.set(host)

    try:
        # Failed validation will throw an error.
        _validate_entra_auth()
    except Exception as e:
        raise MlflowException(f"`mlflow.login()` failed with error: {e}")

    # Instruct the tracking client to authenticate requests via the "entra" request auth
    # provider, which acquires (and transparently refreshes) tokens on each request.
    MLFLOW_TRACKING_AUTH.set("entra")
    _logger.info(f"Successfully signed in with Microsoft Entra ID. Host: {host}.")
    return host
