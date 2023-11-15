import configparser
import getpass
import logging
import os
import subprocess
from typing import NamedTuple, Optional, Tuple

from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException

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


def login(backend="databricks"):
    """Configure MLflow server authentication and connect MLflow to tracking server.

    This method provides a simple way to connect MLflow to its tracking server. Currently only
    Databricks tracking server is supported. Users will be prompted to enter the credentials if no
    existing Databricks profile is found, and the credentials will be saved to `~/.databrickscfg`.

    Args:
        backend: string, the backend of the tracking server. Currently only "databricks" is
            supported.

    .. code-block:: python
        :caption: Example

        import mlflow

        mlflow.login()
        with mlflow.start_run():
            mlflow.log_param("p", 0)
    """
    from mlflow.tracking import set_tracking_uri

    if backend == "databricks":
        _databricks_login()
        set_tracking_uri("databricks")
    else:
        raise MlflowException(
            f"Currently only 'databricks' backend is supported, received `backend={backend}`."
        )


def _validate_databricks_auth():
    """Validate Databricks authentication."""
    max_attempts = 3  # Allow 3 attempts for timeout.
    timeout = 3
    for i in range(max_attempts):
        try:
            # If the host name is invalid, the command will hang.
            # If the credential is invalid, the command will return non-zero exit code.
            # If both host and credential are valid, it will return zero exit code.
            result = subprocess.run(
                ["databricks", "clusters", "list-zones"],
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise MlflowException("Failed to validate databricks credentials.")
        except subprocess.TimeoutExpired:
            if i == max_attempts - 1:
                raise subprocess.TimeoutExpired(
                    "Timeout while signing in Databricks",
                    timeout=timeout,
                )

            _logger.error("Timeout while signing in Databricks, retrying...")


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


def _databricks_login():
    """Set up databricks authentication."""
    try:
        # Failed validation will throw an error.
        _validate_databricks_auth()
        return
    except Exception:
        # If no valid auth is found, we will prompt the user to enter thepy auth.
        pass

    _logger.info("No valid Databricks credentials found, please enter your credentials...")

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
        return
    except subprocess.TimeoutExpired:
        raise MlflowException(
            "Timeout while signing in Databricks, max retry reached. This often happens when you "
            "are using an invalid host. Please check your credentials and retry `mlflow.login`."
        )
    except Exception as e:
        # If user entered invalid auth, we will raise an error and ask users to retry.
        raise MlflowException(f"Failed to sign in Databricks, please retry `mlflow.login()`: {e}")
