import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple

from mlflow.environment_variables import (
    MLFLOW_TRACKING_AUTH,
    MLFLOW_TRACKING_AWS_SIGV4,
    MLFLOW_TRACKING_CLIENT_CERT_PATH,
    MLFLOW_TRACKING_INSECURE_TLS,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_SERVER_CERT_PATH,
    MLFLOW_TRACKING_TOKEN,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds

_logger = logging.getLogger(__name__)


class MlflowCreds(NamedTuple):
    username: Optional[str]
    password: Optional[str]


def _get_credentials_path() -> str:
    return os.path.expanduser("~/.mlflow/credentials")


def _read_mlflow_creds_from_file() -> Tuple[Optional[str], Optional[str]]:
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


def _read_mlflow_creds_from_env() -> Tuple[Optional[str], Optional[str]]:
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

    This method provides a simple way to connect MLflow to its tracking server. Currently only
    Databricks tracking server is supported. Users will be prompted to enter the credentials if no
    existing Databricks profile is found, and the credentials will be saved to `~/.databrickscfg`.

    Args:
        backend: string, the backend of the tracking server. Currently only "databricks" is
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
    else:
        raise MlflowException(
            f"Currently only 'databricks' backend is supported, received `backend={backend}`."
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
