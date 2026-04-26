from typing import Any

from pydantic import BaseModel, Field

import mlflow


class AgentInfo(BaseModel):
    """Metadata describing an agent server's identity and interface.

    Passed to ``AgentServer`` as a constructor argument to enrich the
    ``GET /agent/info`` endpoint with additional discovery information.

    Args:
        name: Server/agent name. Defaults to the ``DATABRICKS_APP_NAME``
            environment variable or ``"mlflow_agent_server"``.
        use_case: The use case category. Defaults to ``"agent"``.
        mlflow_version: The MLflow version running the server.
            Defaults to the installed ``mlflow.__version__``.
        agent_api: The agent API protocol (e.g. ``"responses"``).
            Automatically set when ``agent_type="ResponsesAgent"``.
        description: Human-readable description of what the agent does.
        version: Version string for the agent.
        metadata: Extensible dict for additional information.
            Reserved keys: ``custom_inputs_schema``, ``custom_outputs_schema``.
        tags: Arbitrary key-value metadata for categorization.
    """

    name: str = "mlflow_agent_server"
    use_case: str = "agent"
    mlflow_version: str = Field(default_factory=lambda: mlflow.__version__)
    agent_api: str | None = None
    description: str | None = None
    version: str | None = None
    metadata: dict[str, Any] | None = None
    tags: dict[str, str] | None = None
