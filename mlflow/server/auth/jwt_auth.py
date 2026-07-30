"""Sample JWT authentication module for testing purposes.

NOT SUITABLE FOR PRODUCTION USE.
"""

import configparser
import logging
from pathlib import Path
import traceback
from typing import Union

import jwt
from flask import Response, make_response, request
from mlflow import MlflowException
from werkzeug.datastructures import Authorization

from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH
from mlflow.entities.view_type import ViewType

from sqlalchemy.exc import NoResultFound

from mlflow.server.auth import store, authenticate_request_basic_auth
from mlflow.server.handlers import _get_tracking_store, _get_model_registry_store
from datetime import datetime, timedelta
import requests

BEARER_PREFIX = "bearer "
BASIC_PREFIX  = "basic "
_logger = logging.getLogger(__name__)

_logger.setLevel(logging.DEBUG)

def get_jwt_public_key(url):
        # fetch public key from url
        _logger.debug(f"Fetching public key from {url}")
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            secret = response.json()["public_key"]
            return f"-----BEGIN PUBLIC KEY-----\n{secret}\n-----END PUBLIC KEY-----"
        else:
            _logger.error(f"Failed to obtain jwt public key from {url}")
            raise ValueError(f"Failed to obtain public key from {url}")


def prepare_config():
    config = configparser.ConfigParser()
    _logger.debug(Path(__file__).parent.joinpath("basic_auth.ini").resolve())
    config.read(MLFLOW_AUTH_CONFIG_PATH.get() or Path(__file__).parent.joinpath("basic_auth.ini").resolve())
    if not 'jwt_public_key' in config['mlflow']:
        if 'keycloak_realm_url' in config['mlflow']:
            config['mlflow']['jwt_public_key'] = get_jwt_public_key(config['mlflow']['keycloak_realm_url'])
        else:
            raise ValueError("jwt_public_key or keycloak_realm_url must be provided in the configuration file")
    
    if 'jwt_username_key' not in config['mlflow']:
        config['mlflow']['jwt_username_key'] = 'preferred_username'
    return config

config = prepare_config()
backend_store = _get_tracking_store()
registry_store = _get_model_registry_store()
admin_username = config["mlflow"]["admin_username"]
admin_password = config["mlflow"]["admin_password"]

def user_exists(username):
    try:
        user = store.get_user(username)
        _logger.debug(f"User {username} exists and admin {user.is_admin} and {user.id}")
        return user.id
    except NoResultFound:
        return False
    except Exception as e:
        _logger.warning(f"Failed with {e}")
        return False

def create_user(username, password, is_admin):
    user = store.create_user(username, password, is_admin)
    return user

# obtain higher permission mapped to user roles
def _get_permission_from_roles(user_roles):
    role_hierarchy = ['NO_PERMISSIONS', 'READ', 'EDIT', 'MANAGE', 'ADMIN']
    # Convert role hierarchy to lowercase for comparisons
    role_hierarchy_lower = [role.lower() for role in role_hierarchy]
    highest_role = role_hierarchy[0]
    
    for role, groups in config['role_mappings'].items():
        # Convert groups to lowercase and split
        groups_list = [group.strip().lower() for group in groups.split(',')]
        # Convert user_roles to lowercase for comparison
        user_roles_lower = [role.lower() for role in user_roles]
        
        if any(group in user_roles_lower for group in groups_list):
            if highest_role is None or role_hierarchy_lower.index(role.lower()) > role_hierarchy_lower.index(highest_role.lower()):
                highest_role = role

    return highest_role.upper()

# update permissions of already created experiments and models

def update_experiment_permissions(user, user_id, user_roles, force_update=False):
    _logger.debug(f"Updating permissions for user with role: {user_roles}")
    user_permission = _get_permission_from_roles(user_roles)
    exps = backend_store.search_experiments(ViewType.ACTIVE_ONLY)
    experiment_ids = [ exp.experiment_id for exp in exps]
    for i, exp_id in enumerate(experiment_ids):
        try:
            # check if permission exists, create it if not
            exp_perm = store.get_experiment_permission(exp_id, user)

            # if permission exists, update it if force_update is True and user is not admin
            if force_update and exp_perm._permission != 'MANAGE':
                store.update_experiment_permission(exp_id, user, user_permission)
                _logger.debug(f"Permission for user {user} with {user_id} updated for experiment {exp_id} for {exp_perm._user_id}")
        except MlflowException as e:
            error = f"Experiment permission with experiment_id={exp_id} and username={user} not found"
            # if error is permission doesn't exist, create it, because MLflow sets every permission to default_permission or admin
            if str(e) == error:
                # if experiment is recently created, do nothing
                creation_time = datetime.fromtimestamp(exps[i]._creation_time / 1000)
                if creation_time < datetime.now() - timedelta(minutes=3):
                    store.create_experiment_permission(exp_id, user, user_permission)
                    _logger.debug(f"Permission for user {user} created for experiment {exp_id}")
            else:
                # raise error if another MlflowException is raised
                raise e
        except Exception as e:
            raise e

def update_registry_permissions(user, user_id, user_roles, force_update=False):
    _logger.debug(f"Updating permissions for user with role: {user_roles}")
    user_permission = _get_permission_from_roles(user_roles)
    reg_models = registry_store.search_registered_models()
    reg_model_names = [ model.name for model in reg_models]
    for i, model_name in enumerate(reg_model_names):
        try:
            # check if permission exists, create it if not
            model_perm = store.get_registered_model_permission(model_name, user)
            _logger.debug(f"Permission for user {user} for model {model_name} is {model_perm.__dict__}")
            # if permission exists, update it if force_update is True and user is not admin
            if force_update and model_perm._permission != 'MANAGE':
                store.update_registered_model_permission (model_name, user, user_permission)
                _logger.debug(f"Permission for user {user} with {user_id} updated for registered model {model_name} for {reg_models[i].__dict__}")
        except MlflowException as e:
            error = f"Registered model permission with name={model_name} and username={user} not found"
            # if error is permission doesn't exist, create it, because MLflow sets every permission to default_permission or admin
            if str(e) == error:
                # if registered model is recently created, do nothing
                creation_time = datetime.fromtimestamp(reg_models[i]._creation_time / 1000)
                if creation_time < datetime.now() - timedelta(minutes=3):
                    store.create_registered_model_permission(model_name, user, user_permission)
                    _logger.debug(f"Permission for user {user} created for registered model {model_name}")
            else:
                # raise error if another MlflowException is raised
                raise e
        except Exception as e:
            raise e


def update_permissions(username, user_id, user_roles, force_update=False):
    update_experiment_permissions(username, user_id, user_roles, force_update)
    update_registry_permissions(username, user_id, user_roles, force_update)

def authenticate_sso_request() -> Union[Authorization, Response]:
    _logger.debug("Getting token")
    error_response = make_response()
    error_response.status_code = 401
    error_response.set_data(
        "You are not authenticated. Please provide a valid JWT Bearer token with the request."
    )
    error_response.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'

    token = request.headers.get("authorization")

    if token is not None and token.lower().startswith(BEARER_PREFIX):
        token = token[len(BEARER_PREFIX) :]  # Remove prefix
        try:
            unverified_header = jwt.get_unverified_header(token)
            algorithm = unverified_header.get("alg")
            keycloak_algorithms = ["RS256", "RS384", "RS512", "HS256", "HS384", "HS512", "ES256", "ES384", "ES512"]
            if algorithm not in keycloak_algorithms:
                _logger.error(f"Algorithm {algorithm} not supported")
                return error_response
            audience = config['mlflow'].get('keycloak_client', 'mlflow')
            token_info = jwt.decode(token, config['mlflow']['jwt_public_key'], algorithms=keycloak_algorithms, audience=audience)
            if not token_info:  # pragma: no cover
                _logger.warning("No token_info returned")
                return error_response
            # Check token expiration
            exp = token_info.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                    _logger.error("Token expired")
                    return error_response
            token_info["username"] = token_info[config['mlflow']['jwt_username_key']]

            _logger.debug(f"User {token_info['username']} authenticated")

            # Check if admin for user creation and permission sync
            is_admin = any(group in config['role_mappings']['admin'].split(',') for group in token_info['groups'])
            # Create user if they don't exist
            user_id = user_exists(token_info['username'])
            if not user_id:
                create_user(token_info["username"], token_info['sub'], is_admin)
                _logger.debug(f"User {token_info['username']} created locally")
            
            store.update_user(username=token_info['username'], is_admin=is_admin)
            
            # Update permissions according to defaults
            if not is_admin:
                update_permissions(token_info['username'],user_id, token_info['groups'], force_update=True)

            return Authorization(auth_type="jwt", data=token_info)
        except Exception as e:
            _logger.error(traceback.format_exc())
            _logger.error(e)
    elif token is not None and token.lower().startswith(BASIC_PREFIX):
        auth = authenticate_request_basic_auth()
        if auth.status_code is None:
            return auth
        else:
            _logger.warning("Missing or invalid basic authentication")
            error_response.set_data("You are not authenticated, Basic authentication failed")
            return error_response

    _logger.warning("Missing or invalid authorization token")
    return error_response
