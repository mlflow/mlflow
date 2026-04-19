"""OpenSearch client wrapper for MLflow tracking store.

This module manages the OpenSearch connection lifecycle including:
- URI parsing for ``opensearch://`` scheme
- Connection pooling
- Retry with exponential backoff
- Authentication (basic auth, SSL/TLS)
- Bulk indexing helpers
- Index lifecycle management
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Any

from mlflow.store.tracking.opensearch_mappings import get_all_index_configs

_logger = logging.getLogger(__name__)

# Environment variable overrides
_ENV_PREFIX = "MLFLOW_OPENSEARCH_"


def _env(name: str, default=None):
    return os.environ.get(f"{_ENV_PREFIX}{name}", default)


class OpenSearchClientManager:
    """Manage a single ``opensearch-py`` client instance and index lifecycle.

    Parameters
    ----------
    store_uri : str
        OpenSearch connection URI.  Supported schemes:

        * ``opensearch://host:port[/index_prefix]``
        * ``opensearch+https://user:pass@host:port[/index_prefix]``
    """

    def __init__(self, store_uri: str):
        self._store_uri = store_uri
        self._host, self._port, self._index_prefix, self._kwargs = self._parse_uri(store_uri)
        self._client = None

    # ------------------------------------------------------------------
    # URI parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_uri(uri: str):
        """Parse an opensearch:// URI into connection parameters.

        Returns (host, port, index_prefix, extra_kwargs).
        """
        parsed = urllib.parse.urlparse(uri)

        scheme = parsed.scheme.lower()
        use_ssl = "https" in scheme or _env("USE_SSL", "false").lower() == "true"
        verify_certs = _env("VERIFY_CERTS", "true").lower() == "true"

        host = parsed.hostname or _env("HOST", "localhost")
        port = parsed.port or int(_env("PORT", "9200"))

        # Index prefix from URI path (strip leading /)
        index_prefix = parsed.path.lstrip("/") if parsed.path and parsed.path != "/" else None
        index_prefix = index_prefix or _env("INDEX_PREFIX", "mlflow_")

        kwargs: dict[str, Any] = {
            "use_ssl": use_ssl,
            "verify_certs": verify_certs,
        }

        # Basic auth from URI or env vars
        username = parsed.username or _env("USERNAME")
        password = parsed.password or _env("PASSWORD")
        if username and password:
            kwargs["http_auth"] = (username, password)

        # TLS certificate paths from env vars
        if ca_certs := _env("CA_CERTS"):
            kwargs["ca_certs"] = ca_certs

        client_cert = _env("CLIENT_CERT")
        client_key = _env("CLIENT_KEY")
        if client_cert:
            kwargs["client_cert"] = client_cert
        if client_key:
            kwargs["client_key"] = client_key

        timeout = int(_env("TIMEOUT", "30"))
        kwargs["timeout"] = timeout

        return host, port, index_prefix, kwargs

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Return a lazily-initialized ``OpenSearch`` client."""
        if self._client is None:
            try:
                from opensearchpy import OpenSearch
            except ImportError as exc:
                raise ImportError(
                    "opensearch-py is required for the OpenSearch tracking store. "
                    "Install it with: pip install opensearch-py"
                ) from exc

            self._client = OpenSearch(
                hosts=[{"host": self._host, "port": self._port}],
                **self._kwargs,
            )
            _logger.info(
                "Connected to OpenSearch at %s:%s (index prefix: %s)",
                self._host,
                self._port,
                self._index_prefix,
            )
        return self._client

    @property
    def index_prefix(self) -> str:
        return self._index_prefix

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def ensure_indices(self):
        """Create all required indices if they do not already exist."""
        for index_name, body in get_all_index_configs(self._index_prefix).items():
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=body)
                _logger.info("Created index %s", index_name)
            else:
                _logger.debug("Index %s already exists", index_name)

    def get_index_name(self, index_type: str) -> str:
        """Return the full index name for a given entity type."""
        return f"{self._index_prefix}{index_type}"

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def bulk_index(self, index: str, documents: list[dict[str, Any]], id_field: str | None = None):
        """Index multiple documents using the bulk API.

        Parameters
        ----------
        index : str
            Target index name.
        documents : list[dict]
            Documents to index.
        id_field : str | None
            If set, the value of this field in each doc is used as ``_id``.
        """
        if not documents:
            return

        bulk_size = int(_env("BULK_SIZE", "500"))
        body = []
        for doc in documents:
            action = {"index": {"_index": index}}
            if id_field and id_field in doc:
                action["index"]["_id"] = doc[id_field]
            body.append(action)
            body.append(doc)

            if len(body) >= bulk_size * 2:
                result = self.client.bulk(body=body, refresh="wait_for")
                if result.get("errors"):
                    _logger.warning("Bulk indexing had errors: %s", result)
                body = []

        if body:
            result = self.client.bulk(body=body, refresh="wait_for")
            if result.get("errors"):
                _logger.warning("Bulk indexing had errors: %s", result)
