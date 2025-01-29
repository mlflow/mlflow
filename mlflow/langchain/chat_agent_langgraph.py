import json
import uuid
from typing import Annotated, Any, Generator, Literal, Optional, TypedDict, Union

from langchain_core.messages import AnyMessage
from langchain_core.messages import ToolCall as LCToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from mlflow.langchain.utils.chat import convert_lc_message_to_chat_message
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentParams, ChatAgentResponse
from mlflow.utils.annotations import experimental


def _add_agent_messages(left: list[dict], right: list[dict]):
    # assign missing ids
    for m in left:
        if m.get("id") is None:
            m["id"] = str(uuid.uuid4())
    for m in right:
        if m.get("id") is None:
            m["id"] = str(uuid.uuid4())

    # merge
    left_idx_by_id = {m.get("id"): i for i, m in enumerate(left)}
    merged = left.copy()
    for m in right:
        if (existing_idx := left_idx_by_id.get(m.get("id"))) is not None:
            merged[existing_idx] = m
        else:
            merged.append(m)
    return merged


class ChatAgentState(TypedDict):
    """Helper class to add custom_outputs to the state of a LangGraph agent. `custom_outputs` is
    overwritten if it already exists.
    """

    messages: Annotated[list, _add_agent_messages]
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
    Helper class to make ToolNodes be compatible with ChatAgentState.
    Parse "attachments" and "custom_outputs" keys from the string output of a
    LangGraph tool.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        """
        Wraps the standard ToolNode invoke method with parsing logic for dictionary string outputs
        for both UC function tools and standard LangChain python tools that include keys
        "attachments" and "custom_outputs".
        """
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

    def _inject_tool_args(
        self,
        tool_call: LCToolCall,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        store: BaseStore,
    ) -> LCToolCall:
        """
        Slightly modified version of `_inject_tool_args` that adds the function args from a
        ChatAgentMessage tool call to a format that is compatible with LangChain tool invocation.
        """
        tool_call["name"] = tool_call["function"]["name"]
        tool_call["args"] = json.loads(tool_call["function"]["arguments"])
        return super()._inject_tool_args(tool_call, input, store)

    def _parse_input(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
        store: BaseStore,
    ) -> tuple[list[LCToolCall], Literal["list", "dict"]]:
        """Slightly modified version of the LangGraph ToolNode `_parse_input` method that skips
        verifying the last message is an LangChain AIMessage, allowing the state to be of the
        ChatAgentMessage format.
        """
        if isinstance(input, list):
            input_type = "list"
            message: AnyMessage = input[-1]
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            input_type = "dict"
            message = messages[-1]
        elif messages := getattr(input, self.messages_key, None):
            # Assume dataclass-like state that can coerce from dict
            input_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        tool_calls = [self._inject_tool_args(call, input, store) for call in message["tool_calls"]]
        return tool_calls, input_type


@experimental
class LangGraphChatAgent(ChatAgent):
    """
    Helper class to wrap LangGraph agents as a ChatAgent.
    """

    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict(
        self, model_input: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ):
        response = ChatAgentResponse(messages=[])
        for event in self.agent.stream(
            {"messages": self._convert_messages_to_dict(model_input)}, stream_mode="updates"
        ):
            for node_data in event.values():
                if not node_data:
                    continue
                for msg in node_data.get("messages", []):
                    response.messages.append(ChatAgentMessage(**msg))
                if "custom_outputs" in node_data:
                    response.custom_outputs = node_data["custom_outputs"]
        return response

    # TODO trace this by default once manual tracing of predict_stream is supported
    def predict_stream(
        self, model_input: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ) -> Generator[Any, Any, ChatAgentResponse]:
        for event in self.agent.stream(
            {"messages": self._convert_messages_to_dict(model_input)}, stream_mode="updates"
        ):
            for node_data in event.values():
                if not node_data:
                    continue
                yield ChatAgentResponse(**node_data)
