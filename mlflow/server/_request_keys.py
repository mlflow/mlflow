"""
Shared constants for request-state keys written by auth middleware and read by route handlers.

These form a contract across files; centralising them here prevents mismatches.
"""

# Flask g key: auth narrows the existing experiment_ids field without changing the raw request.
SCOPED_EXPERIMENT_IDS = "mlflow_scoped_experiment_ids"
SCOPED_EXPERIMENT_FILTER = "mlflow_scoped_experiment_filter"

# Starlette request.state keys (FastAPI request scope)
SCOPED_MCP_SERVER_FILTER = "mlflow_scoped_mcp_server_filter"
SCOPED_MCP_ACCESS_ENDPOINT_FILTER = "mlflow_scoped_mcp_access_endpoint_filter"
MCP_SERVER_ALLOWED_ACTIONS = "mcp_server_allowed_actions"
