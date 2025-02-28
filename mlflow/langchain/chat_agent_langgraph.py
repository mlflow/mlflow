import json
import uuid
from typing import Annotated, Any, Optional, TypedDict

try:
    from langchain_core.messages import AnyMessage, BaseMessage, convert_to_messages
    from langchain_core.runnables import RunnableConfig
    from langchain_core.runnables.utils import Input
    from langgraph.prebuilt.tool_node import ToolNode
except ImportError as e:
    raise ImportError(
        "Please install `langchain>=0.2.17` and `langgraph>=0.2.0` to use LangGraph ChatAgent"
        "helpers."
    ) from e

from mlflow.langchain.utils.chat import convert_lc_message_to_chat_message
from mlflow.types.agent import ChatAgentMessage
from mlflow.utils.annotations import experimental


def _add_agent_messages(left: list[dict], right: list[dict]):
    # assign missing ids
    for i, m in enumerate(left):
        if isinstance(m, BaseMessage):
            left[i] = parse_message(m)
        if left[i].get("id") is None:
            left[i]["id"] = str(uuid.uuid4())

    for i, m in enumerate(right):
        if isinstance(m, BaseMessage):
            right[i] = parse_message(m)
        if right[i].get("id") is None:
            right[i]["id"] = str(uuid.uuid4())

    # merge
    left_idx_by_id = {m.get("id"): i for i, m in enumerate(left)}
    merged = left.copy()
    for m in right:
        if (existing_idx := left_idx_by_id.get(m.get("id"))) is not None:
            merged[existing_idx] = m
        else:
            merged.append(m)
    return merged


@experimental
class ChatAgentState(TypedDict):
    """
    Helper class that enables building a LangGraph agent that produces ChatAgent-compatible
    messages as state is updated. Other ChatAgent request fields (custom_inputs, context) and
    response fields (custom_outputs) are also exposed within the state so they can be used and
    updated over the course of agent execution. Use this class with
    :py:class:`ChatAgentToolNode <mlflow.langchain.chat_agent_langgraph.ChatAgentToolNode>`.

    **LangGraph ChatAgent Example**

    This example has been tested to work with LangGraph 0.2.70.

    Step 1: Create the LangGraph Agent

    This example is adapted from LangGraph's
    `create_react_agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`__
    documentation. The notable differences are changes to be ChatAgent compatible. They include:

    - We use :py:class:`ChatAgentState <mlflow.langchain.chat_agent_langgraph.ChatAgentState>`,
      which has an internal state of
      :py:class:`ChatAgentMessage <mlflow.types.agent.ChatAgentMessage>`
      objects and a ``custom_outputs`` attribute under the hood
    - We use :py:class:`ChatAgentToolNode <mlflow.langchain.chat_agent_langgraph.ChatAgentToolNode>`
      instead of LangGraph's ToolNode to enable returning attachments and custom_outputs from
      LangChain and UnityCatalog Tools

    .. code-block:: python

        from typing import Optional, Sequence, Union

        from langchain_core.language_models import LanguageModelLike
        from langchain_core.runnables import RunnableConfig, RunnableLambda
        from langchain_core.tools import BaseTool
        from langgraph.graph import END, StateGraph
        from langgraph.graph.graph import CompiledGraph
        from langgraph.prebuilt.tool_executor import ToolExecutor
        from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode


        def create_tool_calling_agent(
            model: LanguageModelLike,
            tools: Union[ToolExecutor, Sequence[BaseTool]],
            agent_prompt: Optional[str] = None,
        ) -> CompiledGraph:
            model = model.bind_tools(tools)

            def routing_logic(state: ChatAgentState):
                last_message = state["messages"][-1]
                if last_message.get("tool_calls"):
                    return "continue"
                else:
                    return "end"

            if agent_prompt:
                system_message = {"role": "system", "content": agent_prompt}
                preprocessor = RunnableLambda(
                    lambda state: [system_message] + state["messages"]
                )
            else:
                preprocessor = RunnableLambda(lambda state: state["messages"])
            model_runnable = preprocessor | model

            def call_model(
                state: ChatAgentState,
                config: RunnableConfig,
            ):
                response = model_runnable.invoke(state, config)

                return {"messages": [response]}

            workflow = StateGraph(ChatAgentState)

            workflow.add_node("agent", RunnableLambda(call_model))
            workflow.add_node("tools", ChatAgentToolNode(tools))

            workflow.set_entry_point("agent")
            workflow.add_conditional_edges(
                "agent",
                routing_logic,
                {
                    "continue": "tools",
                    "end": END,
                },
            )
            workflow.add_edge("tools", "agent")

            return workflow.compile()

    Step 2: Define the LLM and your tools

    If you want to return attachments and custom_outputs from your tool, you can return a
    dictionary with keys “content”, “attachments”, and “custom_outputs”. This dictionary will be
    parsed out by the ChatAgentToolNode and properly stored in your LangGraph's state.


    .. code-block:: python

        from random import randint
        from typing import Any

        from databricks_langchain import ChatDatabricks
        from langchain_core.tools import tool


        @tool
        def generate_random_ints(min: int, max: int, size: int) -> dict[str, Any]:
            \"""Generate size random ints in the range [min, max].\"""
            attachments = {"min": min, "max": max}
            custom_outputs = [randint(min, max) for _ in range(size)]
            content = f"Successfully generated array of {size} random ints in [{min}, {max}]."
            return {
                "content": content,
                "attachments": attachments,
                "custom_outputs": {"random_nums": custom_outputs},
            }


        mlflow.langchain.autolog()
        tools = [generate_random_ints]
        llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
        langgraph_agent = create_tool_calling_agent(llm, tools)


    Step 3: Wrap your LangGraph agent with ChatAgent

    This makes your agent easily loggable and deployable with the PyFunc flavor in serving.

    .. code-block:: python

        from typing import Any, Generator, Optional

        from langgraph.graph.state import CompiledStateGraph
        from mlflow.pyfunc import ChatAgent
        from mlflow.types.agent import (
            ChatAgentChunk,
            ChatAgentMessage,
            ChatAgentResponse,
            ChatContext,
        )


        class LangGraphChatAgent(ChatAgent):
            def __init__(self, agent: CompiledStateGraph):
                self.agent = agent

            def predict(
                self,
                messages: list[ChatAgentMessage],
                context: Optional[ChatContext] = None,
                custom_inputs: Optional[dict[str, Any]] = None,
            ) -> ChatAgentResponse:
                request = {"messages": self._convert_messages_to_dict(messages)}

                messages = []
                for event in self.agent.stream(request, stream_mode="updates"):
                    for node_data in event.values():
                        messages.extend(
                            ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                        )
                return ChatAgentResponse(messages=messages)

            def predict_stream(
                self,
                messages: list[ChatAgentMessage],
                context: Optional[ChatContext] = None,
                custom_inputs: Optional[dict[str, Any]] = None,
            ) -> Generator[ChatAgentChunk, None, None]:
                request = {"messages": self._convert_messages_to_dict(messages)}
                for event in self.agent.stream(request, stream_mode="updates"):
                    for node_data in event.values():
                        yield from (
                            ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                        )


        chat_agent = LangGraphChatAgent(langgraph_agent)

    Step 4: Test out your model

    Call ``.predict()`` and ``.predict_stream`` with dictionaries with the ChatAgentRequest schema.

    .. code-block:: python

        chat_agent.predict({"messages": [{"role": "user", "content": "What is 10 + 10?"}]})

        for event in chat_agent.predict_stream(
            {"messages": [{"role": "user", "content": "Generate me a few random nums"}]}
        ):
            print(event)

    This LangGraph ChatAgent can be logged with the logging code described in the "Logging a
    ChatAgent" section of the docstring of :py:class:`ChatAgent <mlflow.pyfunc.ChatAgent>`.
    """

    messages: Annotated[list, _add_agent_messages]
    context: Optional[dict[str, Any]]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def parse_message(
    msg: AnyMessage, name: Optional[str] = None, attachments: Optional[dict] = None
) -> dict[str, Any]:
    """
    Parse different LangChain message types into their ChatAgentMessage schema dict equivalents
    """
    chat_message_dict = convert_lc_message_to_chat_message(msg).model_dump_compat()
    chat_message_dict["attachments"] = attachments
    chat_message_dict["name"] = msg.name or name
    chat_message_dict["id"] = msg.id
    # _convert_to_message from langchain_core.messages.utils expects an empty string instead of None
    if not chat_message_dict.get("content"):
        chat_message_dict["content"] = ""

    chat_agent_msg = ChatAgentMessage(**chat_message_dict)
    return chat_agent_msg.model_dump_compat(exclude_none=True)


@experimental
class ChatAgentToolNode(ToolNode):
    """
    Helper class to make ToolNodes be compatible with
    :py:class:`ChatAgentState <mlflow.langchain.chat_agent_langgraph.ChatAgentState>`.
    Parse ``attachments`` and ``custom_outputs`` keys from the string output of a
    LangGraph tool.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        """
        Wraps the standard ToolNode invoke method to:
        - Parse ChatAgentState into LangChain messages
        - Parse dictionary string outputs from both UC function and standard LangChain python tools
          that include keys ``content``, ``attachments``, and ``custom_outputs``.
        """
        messages = input["messages"]
        for msg in messages:
            for tool_call in msg.get("tool_calls", []):
                tool_call["name"] = tool_call["function"]["name"]
                tool_call["args"] = json.loads(tool_call["function"]["arguments"])
        input["messages"] = convert_to_messages(messages)

        result = super().invoke(input, config, **kwargs)

        messages = []
        custom_outputs = None
        for m in result["messages"]:
            try:
                return_obj = json.loads(m.content)
                if all(key in return_obj for key in ("format", "value", "truncated")):
                    # Dictionary output with custom_outputs and attachments from a UC function
                    try:
                        return_obj = json.loads(return_obj["value"])
                    except Exception:
                        pass
                if "custom_outputs" in return_obj:
                    custom_outputs = return_obj["custom_outputs"]
                messages.append(parse_message(m, attachments=return_obj.get("attachments")))
            except Exception:
                messages.append(parse_message(m))
        return {"messages": messages, "custom_outputs": custom_outputs}
