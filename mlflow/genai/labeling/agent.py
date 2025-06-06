from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from databricks.agents.review_app.labeling import Agent as AgentsAgent


class Agent:
    """A wrapper around the databricks.agents.review_app.labeling.Agent class."""

    def __init__(self, agent: "AgentsAgent"):
        self._agent = agent

    @property
    def agent_id(self) -> Optional[str]:
        """The unique identifier of the agent."""
        return self._agent.agent_id

    @property
    def name(self) -> Optional[str]:
        """The name of the agent."""
        return self._agent.name

    @property
    def description(self) -> Optional[str]:
        """The description of the agent."""
        return self._agent.description

    @property
    def model_id(self) -> Optional[str]:
        """The model ID used by the agent."""
        return self._agent.model_id

    @property
    def create_time(self) -> Optional[str]:
        """The time the agent was created."""
        return self._agent.create_time

    @property
    def created_by(self) -> Optional[str]:
        """The user who created the agent."""
        return self._agent.created_by

    @property
    def last_update_time(self) -> Optional[str]:
        """The time the agent was last updated."""
        return self._agent.last_update_time

    @property
    def last_updated_by(self) -> Optional[str]:
        """The user who last updated the agent."""
        return self._agent.last_updated_by
