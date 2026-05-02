import os
from typing import Any

from pydantic import BaseModel, Field

from mlflow.version import VERSION


class AgentInfo(BaseModel):
    """Metadata describing an agent server's identity and interface.

    Passed to ``AgentServer`` as a constructor argument to enrich the
    ``GET /agent/info`` endpoint with additional discovery information.

    Args:
        name: Server/agent name. Defaults to the ``DATABRICKS_APP_NAME``
            environment variable, or ``"mlflow_agent_server"`` if it is unset.
        use_case: The use case category. Defaults to ``"agent"``.
        mlflow_version: The MLflow version running the server.
            Defaults to the installed MLflow version.
        agent_api: The agent API protocol (e.g. ``"responses"``).
            Automatically set when ``agent_type="ResponsesAgent"``.
        description: Human-readable description of what the agent does.
        version: Version string for the agent.
        metadata: Extensible dict for additional information. By convention,
            ``custom_inputs_schema`` and ``custom_outputs_schema`` are used to
            advertise the agent's custom input/output JSON schemas.
    """

    name: str = Field(
        default_factory=lambda: os.environ.get("DATABRICKS_APP_NAME", "mlflow_agent_server")
    )
    use_case: str = "agent"
    mlflow_version: str = Field(default_factory=lambda: VERSION)
    agent_api: str | None = None
    description: str | None = None
    version: str | None = None
    metadata: dict[str, Any] | None = None
