import configparser
from pathlib import Path
from typing import NamedTuple

from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH


class AuthConfig(NamedTuple):
    default_permission: str
    database_uri: str
    admin_username: str
    admin_password: str
    jwt_public_key: str
    jwt_username_key: str
    authorization_function: str


def _get_auth_config_path() -> str:
    return (
        MLFLOW_AUTH_CONFIG_PATH.get() or Path(__file__).parent.joinpath("basic_auth.ini").resolve()
    )


secret = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvHFqHaYmUFQq/x0oo1cw20q0YZoCfrKGd7gvhNX1yJZGN7q+2+lUnYdwBHmXubB3R4lA0a9ggY9oMkMebKHYtRfzQZAu2NZD8y/IqdPnN+fI4DdEK8I5mCrXkdyEfaW1oqhfjGP0PmyBHYUblAcEkgl6y4Kp8+bS9IkJOfcTUbWKlEv86y6ktt/7bzIqKdTu9qBPHmfAGABAyzO1HKBH+RxvjY3F91MrSRXRoIdl8mhshtDhEfOt0LWCXz9rCROb23ybKlCQAuwkK8dhzwq5/9aaxksqXVRkAdEqK6IA5JyGcqoUPSsVv01wQAzIjpH5tvGrUS07TIVeE3QHIEtPTwIDAQAB"
public_key=f"-----BEGIN PUBLIC KEY-----\n{secret}\n-----END PUBLIC KEY-----"

def read_auth_config() -> AuthConfig:
    config_path = _get_auth_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return AuthConfig(
        default_permission=config["mlflow"].get("default_permission", "READ"),
        database_uri=config["mlflow"]["database_uri"],
        admin_username=config["mlflow"]["admin_username"],
        admin_password=config["mlflow"]["admin_password"],
        jwt_public_key=config["mlflow"].get("jwt_public_key",public_key),
        jwt_username_key=config["mlflow"].get("jwt_username_key","username"),
        authorization_function=config["mlflow"].get(
            "authorization_function", "mlflow.server.auth:authenticate_request_basic_auth"
        ),
    )
