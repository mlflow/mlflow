"""
Databricks SQL Store implementation that extends RestStore for SQL-based operations.

This store provides direct Databricks SQL query capabilities while delegating
standard tracking operations to the parent RestStore.
"""

import datetime
import json
import logging
import os
import time
from decimal import Decimal
from functools import partial

import requests

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.databricks_utils import get_databricks_host_creds

_logger = logging.getLogger(__name__)


class DatabricksSqlStore(RestStore):
    """
    Databricks SQL Store that extends RestStore with SQL query capabilities.

    This class provides core Databricks SQL functionality including:
    - Spark session management
    - SQL query execution
    - API access to Databricks services
    """

    def __init__(self, store_uri="databricks"):
        """
        Initialize DatabricksSqlStore with Databricks connection.

        Args:
            store_uri: Databricks URI (e.g., 'databricks' or 'databricks://<profile>')
        """
        # Initialize RestStore with Databricks credentials
        super().__init__(partial(get_databricks_host_creds, store_uri))
        self._spark_session = None

    def __del__(self):
        """Close Spark session when store is deleted."""
        self.close_spark_session()

    def close_spark_session(self):
        """Close the Spark session if it exists."""
        if self._spark_session:
            try:
                self._spark_session.stop()
            except:
                pass
            finally:
                self._spark_session = None

    def _get_or_create_spark_session(self):
        """
        Get or create a Spark session for Databricks SQL queries.

        Returns:
            A DatabricksSession configured for serverless compute.
        """
        if self._spark_session is None or not self._is_spark_session_healthy():
            self._create_new_spark_session()
        return self._spark_session

    def _is_spark_session_healthy(self):
        """
        Check if the current Spark session is healthy by running a simple query.

        Returns:
            bool: True if the session is healthy, False otherwise
        """
        if self._spark_session is None:
            return False

        try:
            # Simple health check query
            self._spark_session.sql("SELECT 1").collect()
            return True
        except Exception as e:
            _logger.warning(f"Spark session health check failed: {e}")
            return False

    def _create_new_spark_session(self):
        """
        Create a new Spark session, cleaning up the old one if it exists.
        """
        # Clean up existing session
        if self._spark_session is not None:
            try:
                self._spark_session.stop()
            except Exception as e:
                _logger.warning(f"Error stopping previous Spark session: {e}")
            finally:
                self._spark_session = None

        try:
            # Lazy import to avoid issues when databricks.connect is not installed
            from databricks.connect import DatabricksSession

            # Use environment variables for configuration
            self._spark_session = DatabricksSession.builder.serverless(True).getOrCreate()
            _logger.debug("Created new Databricks Spark session")
        except ImportError as e:
            raise MlflowException(
                "databricks.connect is not installed. Please install it with: "
                "pip install databricks-connect",
                error_code=INVALID_PARAMETER_VALUE,
            ) from e
        except Exception as e:
            raise MlflowException(
                f"Failed to create Databricks session: {e}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def execute_sql(self, query: str):
        """
        Execute a SQL query against Databricks using Spark.

        Args:
            query: The SQL query string to execute

        Returns:
            List of dictionaries representing the query results
        """
        try:
            spark = self._get_or_create_spark_session()
            result = spark.sql(query).collect()

            def convert_value(value):
                """Recursively convert Spark SQL types to JSON-serializable types."""
                if value is None:
                    return None
                elif isinstance(value, (datetime.datetime, datetime.date)):
                    return value.isoformat()
                elif isinstance(value, Decimal):
                    return float(value)
                elif hasattr(value, "asDict"):  # Check for Row-like objects
                    # Convert nested Row objects to dicts
                    return {k: convert_value(v) for k, v in value.asDict().items()}
                elif isinstance(value, dict):
                    # Handle maps
                    return {k: convert_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    # Handle arrays
                    return [convert_value(v) for v in value]
                else:
                    return value

            # Convert rows to dictionaries with proper type conversion
            return [
                {
                    k: convert_value(v)
                    for k, v in (row.asDict() if hasattr(row, "asDict") else row).items()
                }
                for row in result
            ]
        except Exception as e:
            raise MlflowException(
                f"Failed to execute SQL query: {e}", error_code=INVALID_PARAMETER_VALUE
            )

    def _get_trace_table_for_experiment(self, experiment_id: str) -> str | None:
        """
        Get the trace archive table name for an experiment from Databricks monitors API.

        This method calls the /api/2.0/managed-evals/monitors endpoint to retrieve
        monitor information for an experiment, and extracts the trace_archive_table if available.

        Args:
            experiment_id: The experiment ID to get the trace table for

        Returns:
            The trace table name (without backticks) if found, None otherwise
        """
        try:
            # Get host credentials for making the API call
            host_creds = self.get_host_creds()

            # Prepare request
            url = f"{host_creds.host}/api/2.0/managed-evals/monitors"
            headers = {"Content-Type": "application/json"}
            if host_creds.token:
                headers["Authorization"] = f"Bearer {host_creds.token}"

            request_body = json.dumps({"experiment_id": experiment_id})

            _logger.debug(f"Calling GetMonitor API for experiment {experiment_id}: {url}")

            # Make the API call
            resp = requests.post(
                url,
                data=request_body,
                headers=headers,
                verify=not getattr(host_creds, "ignore_tls_verification", False),
            )

            _logger.debug(f"GetMonitor API response status: {resp.status_code}")
            if resp.status_code != 200:
                _logger.warning(
                    f"GetMonitor API failed with status {resp.status_code}: {resp.text}"
                )
                # API call failed, return None to fall back to default behavior
                return None

            response_json = resp.json()
            _logger.debug(f"GetMonitor API response: {response_json}")

            # Extract trace_archive_table from response
            monitor_infos = response_json.get("monitor_infos", [])
            if monitor_infos and len(monitor_infos) > 0:
                monitor = monitor_infos[0].get("monitor", {})
                trace_table = monitor.get("trace_archive_table")
                _logger.debug(f"Found trace_archive_table: {trace_table}")

                if trace_table:
                    # Remove backticks from table name if present
                    return trace_table.replace("`", "")

            return None

        except Exception as e:
            _logger.error(f"Error getting trace table for experiment: {e}")
            # If anything goes wrong, return None
            return None
