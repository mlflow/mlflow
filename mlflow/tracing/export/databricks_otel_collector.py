"""Databricks OTLP collector transport for Unity Catalog trace destinations.

This module provides the transport used by the tracing router. The router owns
destination eligibility, batching, metadata export, and delivery failure
classification. This transport serializes a span batch and sends it to the
Databricks OTLP collector when its client can be initialized.

The transport and its client are initialized without network I/O. Credential,
workspace, and endpoint discovery happen on the first span batch; see
``databricks_otel_client.DatabricksOTelClient`` for the low-level wire
contract.
"""

from typing import Sequence

import requests

from mlflow.entities.span import Span
from mlflow.tracing.export.databricks_otel_client import DatabricksOTelClient
from mlflow.tracing.utils.otlp import build_otlp_export_request


class DatabricksOtelSerializationError(RuntimeError):
    """An OTLP span batch could not be serialized for collector delivery."""


class DatabricksOtelCollectorSpanExporter:
    """Serialize and send span batches to the Databricks OTLP collector."""

    def __init__(
        self,
        tracking_uri: str | None,
        token_source,
        table_name: str,
        host: str | None = None,
        workspace_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self._client = DatabricksOTelClient(
            tracking_uri=tracking_uri,
            token_source=token_source,
            table_name=table_name,
            host=host,
            workspace_id=workspace_id,
            client_id=client_id,
            client_secret=client_secret,
            endpoint=endpoint,
        )

    @property
    def config_warned(self) -> bool:
        """Whether the client emitted a qualified configuration warning."""
        return self._client.config_warned

    @property
    def host(self) -> str | None:
        """The Databricks workspace host after lazy credential discovery."""
        return self._client.host

    @property
    def workspace_id(self) -> str:
        """The Databricks workspace ID after lazy credential discovery."""
        return self._client.workspace_id

    def send_batch(self, spans: Sequence[Span]) -> requests.Response | None:
        """Serialize and send a batch, returning ``None`` when unavailable."""
        if not self._client.ensure_ready():
            return None

        try:
            request = build_otlp_export_request(list(spans))
        except Exception as exc:
            raise DatabricksOtelSerializationError(
                "Failed to build the Databricks OTel collector span export request"
            ) from exc

        try:
            payload = request.SerializeToString()
        except Exception as exc:
            raise DatabricksOtelSerializationError(
                "Failed to serialize the Databricks OTel collector span export request"
            ) from exc

        return self._client.post(payload)

    def close(self) -> None:
        """Close the client's HTTP connection pool."""
        self._client.close()
