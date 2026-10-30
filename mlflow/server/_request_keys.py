"""
Shared constants for request-state keys written by auth middleware and read by route handlers.

These form a contract across files; centralising them here prevents mismatches.
"""

# Flask g key: copy of the authorized experiment_ids scope for response enforcement.
SCOPED_EXPERIMENT_IDS = "mlflow_scoped_experiment_ids"
# Flask g key: effective filter retained for scoped-query error reporting.
SCOPED_EXPERIMENT_FILTER = "mlflow_scoped_experiment_filter"

# Starlette request.state keys (FastAPI request scope)
SCOPED_MCP_SERVER_FILTER = "mlflow_scoped_mcp_server_filter"
SCOPED_MCP_ACCESS_ENDPOINT_FILTER = "mlflow_scoped_mcp_access_endpoint_filter"
MCP_SERVER_ALLOWED_ACTIONS = "mcp_server_allowed_actions"
