from typing import TYPE_CHECKING, Any, Iterable, Union

from mlflow.entities import Trace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    import pandas as pd
    from databricks.agents.review_app import (
        LabelSchema as _LabelSchema,
    )
    from databricks.agents.review_app import (
        ReviewApp as _ReviewApp,
    )
    from databricks.agents.review_app.labeling import Agent as _Agent


class Agent:
    """The agent configuration, used for generating responses in the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, agent: "_Agent"):
        self._agent = agent

    @property
    def agent_name(self) -> str:
        """The name of the agent."""
        return self._agent.agent_name

    @property
    def model_serving_endpoint(self) -> str:
        """The model serving endpoint used by the agent."""
        return self._agent.model_serving_endpoint


class LabelingSession:
    """A session for labeling items in the review app.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(
        self,
        *,
        name: str,
        assigned_users: list[str],
        agent: str | None,
        label_schemas: list[str],
        labeling_session_id: str,
        mlflow_run_id: str,
        review_app_id: str,
        experiment_id: str,
        url: str,
        enable_multi_turn_chat: bool,
        custom_inputs: dict[str, Any] | None,
    ):
        self._name = name
        self._assigned_users = assigned_users
        self._agent = agent
        self._label_schemas = label_schemas
        self._labeling_session_id = labeling_session_id
        self._mlflow_run_id = mlflow_run_id
        self._review_app_id = review_app_id
        self._experiment_id = experiment_id
        self._url = url
        self._enable_multi_turn_chat = enable_multi_turn_chat
        self._custom_inputs = custom_inputs

    @property
    def name(self) -> str:
        """The name of the labeling session."""
        return self._name

    @property
    def assigned_users(self) -> list[str]:
        """The users assigned to label items in the session."""
        return self._assigned_users

    @property
    def agent(self) -> str | None:
        """The agent used to generate responses for the items in the session."""
        return self._agent

    @property
    def label_schemas(self) -> list[str]:
        """The label schemas used in the session."""
        return self._label_schemas

    @property
    def labeling_session_id(self) -> str:
        """The unique identifier of the labeling session."""
        return self._labeling_session_id

    @property
    def mlflow_run_id(self) -> str:
        """The MLflow run ID associated with the session."""
        return self._mlflow_run_id

    @property
    def review_app_id(self) -> str:
        """The review app ID associated with the session."""
        return self._review_app_id

    @property
    def experiment_id(self) -> str:
        """The experiment ID associated with the session."""
        return self._experiment_id

    @property
    def url(self) -> str:
        """The URL of the labeling session in the review app."""
        return self._url

    @property
    def enable_multi_turn_chat(self) -> bool:
        """Whether multi-turn chat is enabled for the session."""
        return self._enable_multi_turn_chat

    @property
    def custom_inputs(self) -> dict[str, Any] | None:
        """Custom inputs used in the session."""
        return self._custom_inputs

    def _get_store(self):
        """
        Get a labeling store instance.

        This method is defined in order to avoid circular imports.
        """
        from mlflow.genai.labeling.stores import _get_labeling_store

        return _get_labeling_store()

    def add_dataset(
        self, dataset_name: str, record_ids: list[str] | None = None
    ) -> "LabelingSession":
        """Add a dataset to the labeling session.

        .. note::
            This functionality is only available in Databricks. Please run
            `pip install mlflow[databricks]` to use it.

        Args:
            dataset_name: The name of the dataset.
            record_ids: Optional. The individual record ids to be added to the session. If not
                provided, all records in the dataset will be added.

        Returns:
            LabelingSession: The updated labeling session.
        """
        store = self._get_store()
        return store.add_dataset_to_session(self, dataset_name, record_ids)

    def add_traces(
        self,
        traces: Union[Iterable[Trace], Iterable[str], "pd.DataFrame"],
    ) -> "LabelingSession":
        """Add traces to the labeling session.

        .. note::
            This functionality is only available in Databricks. Please run
            `pip install mlflow[databricks]` to use it.

        Args:
            traces: Can be either:
                a) a pandas DataFrame with a 'trace' column. The 'trace' column should contain
                either `mlflow.entities.Trace` objects or their json string representations.
                b) an iterable of `mlflow.entities.Trace` objects.
                c) an iterable of json string representations of `mlflow.entities.Trace` objects.

        Returns:
            LabelingSession: The updated labeling session.
        """
        import pandas as pd

        if isinstance(traces, pd.DataFrame):
            if "trace" not in traces.columns:
                raise MlflowException(
                    "traces must have a 'trace' column like the result of mlflow.search_traces()",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            traces = traces["trace"].to_list()

        trace_list: list[Trace] = []
        for trace in traces:
            if isinstance(trace, str):
                trace_list.append(Trace.from_json(trace))
            elif isinstance(trace, Trace):
                trace_list.append(trace)
            elif trace is None:
                raise MlflowException(
                    "trace cannot be None. Must be mlflow.entities.Trace or its json string "
                    "representation.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            else:
                raise MlflowException(
                    f"Expected mlflow.entities.Trace or json string, got {type(trace).__name__}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        store = self._get_store()
        return store.add_traces_to_session(self, trace_list)

    def sync(self, to_dataset: str) -> None:
        """Sync the traces and expectations from the labeling session to a dataset.

        .. note::
            This functionality is only available in Databricks. Please run
            `pip install mlflow[databricks]` to use it.

        Args:
            to_dataset: The name of the dataset to sync traces and expectations to.
        """
        store = self._get_store()
        return store.sync_session_expectations(self, to_dataset)

    def set_assigned_users(self, assigned_users: list[str]) -> "LabelingSession":
        """Set the assigned users for the labeling session.

        .. note::
            This functionality is only available in Databricks. Please run
            `pip install mlflow[databricks]` to use it.

        Args:
            assigned_users: The list of users to assign to the session.

        Returns:
            LabelingSession: The updated labeling session.
        """
        store = self._get_store()
        return store.set_session_assigned_users(self, assigned_users)


class ReviewApp:
    """A review app is used to collect feedback from stakeholders for a given experiment.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    def __init__(self, app: "_ReviewApp"):
        self._app = app

    @property
    def review_app_id(self) -> str:
        """The ID of the review app."""
        return self._app.review_app_id

    @property
    def experiment_id(self) -> str:
        """The ID of the experiment."""
        return self._app.experiment_id

    @property
    def url(self) -> str:
        """The URL of the review app for stakeholders to provide feedback."""
        return self._app.url

    @property
    def agents(self) -> list[Agent]:
        """The agents to be used to generate responses."""
        return [Agent(agent) for agent in self._app.agents]

    @property
    def label_schemas(self) -> list["_LabelSchema"]:
        """The label schemas to be used in the review app."""
        return self._app.label_schemas

    def add_agent(
        self, *, agent_name: str, model_serving_endpoint: str, overwrite: bool = False
    ) -> "ReviewApp":
        """Add an agent to the review app to be used to generate responses.

        .. note::
            This functionality is only available in Databricks. Please run
            `pip install mlflow[databricks]` to use it.

        Args:
            agent_name: The name of the agent.
            model_serving_endpoint: The model serving endpoint to be used by the agent.
            overwrite: Whether to overwrite an existing agent with the same name.

        Returns:
            ReviewApp: The updated review app.
        """
        return ReviewApp(
            self._app.add_agent(
                agent_name=agent_name,
                model_serving_endpoint=model_serving_endpoint,
                overwrite=overwrite,
            )
        )

    def remove_agent(self, agent_name: str) -> "ReviewApp":
        """Remove an agent from the review app.

        .. note::
            This functionality is only available in Databricks. Please run
            `pip install mlflow[databricks]` to use it.

        Args:
            agent_name: The name of the agent to remove.

        Returns:
            ReviewApp: The updated review app.
        """
        return ReviewApp(self._app.remove_agent(agent_name))
