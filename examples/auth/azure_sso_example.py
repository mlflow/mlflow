#!/usr/bin/env python3
"""
Example: Using MLflow with Azure SSO Authentication

This script demonstrates how to:
1. Authenticate with MLflow server using Azure AD credentials
2. Log models and metrics
3. Handle authentication errors

Prerequisites:
    - MLflow server running with Azure SSO enabled
    - Valid Azure AD credentials
    - Environment variables set:
        - MLFLOW_TRACKING_URI: http://localhost:5000
        - AZURE_TENANT_ID: your-tenant-id
        - AZURE_CLIENT_ID: your-client-id
        - AZURE_CLIENT_SECRET: your-client-secret
"""

import os
import mlflow
from mlflow.server import get_app_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_auth():
    """Example 1: Basic authentication with username/password"""
    logger.info("Example 1: Basic Authentication")

    # Set credentials for MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "user@tenant.onmicrosoft.com"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-azure-access-token"

    # Set tracking URI
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Create experiment
        experiment = mlflow.set_experiment("azure_sso_example")
        logger.info(f"Created experiment: {experiment.name}")

        # Start a run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("batch_size", 32)

            # Log metrics
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("loss", 0.05)

            logger.info("Successfully logged metrics to MLflow")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def example_with_permissions():
    """Example 2: Managing permissions with Azure SSO"""
    logger.info("Example 2: Managing Permissions")

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    # Get the auth client
    client = get_app_client("basic-auth", tracking_uri)

    try:
        # Set credentials
        os.environ["MLFLOW_TRACKING_USERNAME"] = "admin@tenant.onmicrosoft.com"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "admin-token"

        # Create experiment
        experiment = mlflow.set_experiment("permissions_example")
        experiment_id = experiment.experiment_id

        # Grant permissions to another user
        other_user = "colleague@tenant.onmicrosoft.com"
        client.create_experiment_permission(
            experiment_id=str(experiment_id),
            username=other_user,
            permission="EDIT"  # EDIT, READ, or MANAGE
        )

        logger.info(f"Granted EDIT permission to {other_user}")

        # List permissions
        perms = client.get_experiment_permissions(str(experiment_id))
        for perm in perms:
            logger.info(f"User {perm.username}: {perm.permission}")

    except Exception as e:
        logger.error(f"Error managing permissions: {str(e)}")
        raise


def example_model_logging():
    """Example 3: Logging models with Azure SSO"""
    logger.info("Example 3: Model Logging")

    # Set credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = "user@tenant.onmicrosoft.com"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "user-token"

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Create a simple model
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Generate dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("solver", "auto")
            mlflow.log_metric("train_score", 0.99)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            logger.info("Model logged successfully")

    except Exception as e:
        logger.error(f"Error logging model: {str(e)}")
        raise


def example_error_handling():
    """Example 4: Handling authentication errors"""
    logger.info("Example 4: Error Handling")

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    # Example 1: Invalid credentials
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "invalid@tenant.onmicrosoft.com"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "invalid-token"

        mlflow.set_experiment("test")
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")

    # Example 2: Expired token
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "user@tenant.onmicrosoft.com"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "expired-token"

        mlflow.set_experiment("test")
    except Exception as e:
        logger.error(f"Token expired: {str(e)}")

    # Example 3: Insufficient permissions
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "readonly@tenant.onmicrosoft.com"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "readonly-token"

        with mlflow.start_run():
            mlflow.log_metric("metric", 1)
    except Exception as e:
        logger.error(f"Permission denied: {str(e)}")


def example_token_refresh():
    """Example 5: Handling token refresh"""
    logger.info("Example 5: Token Refresh")

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # In production, implement token refresh logic
        # This is a simplified example

        def get_fresh_token():
            """Get a fresh token from Azure AD"""
            import requests
            from msal import ConfidentialClientApplication

            tenant_id = os.getenv("AZURE_TENANT_ID")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")

            app = ConfidentialClientApplication(
                client_id=client_id,
                authority=f"https://login.microsoftonline.com/{tenant_id}",
                client_credential=client_secret,
            )

            result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )

            if "access_token" in result:
                return result["access_token"]
            else:
                raise Exception(f"Failed to get token: {result.get('error_description')}")

        # Get fresh token
        token = get_fresh_token()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("AZURE_CLIENT_ID")

        mlflow.set_experiment("token_refresh_example")
        logger.info("Successfully used refreshed token")

    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")


if __name__ == "__main__":
    logger.info("MLflow Azure SSO Examples")
    logger.info("=" * 50)

    try:
        # Uncomment the example you want to run
        example_basic_auth()
        # example_with_permissions()
        # example_model_logging()
        # example_error_handling()
        # example_token_refresh()

    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        exit(1)
