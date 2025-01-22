import json
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

from langchain_core.messages import AnyMessage
from langchain_core.messages import ToolCall as LCToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from mlflow.langchain.utils.chat import convert_lc_message_to_chat_message
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentParams, ChatAgentResponse


def add_agent_messages(left: list[dict], right: list[dict]):
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
    """The state of the agent."""

    messages: Annotated[list, add_agent_messages]
    custom_outputs: dict[str, Any]


@dataclass
class SystemMessage(ChatAgentMessage):
    role: Literal["system"] = field(default="system")


def parse_message(
    msg: AnyMessage, name: Optional[str] = None, attachments: Optional[dict] = None
) -> dict:
    """Parse different message types into their string representations"""
    chat_message_dict = convert_lc_message_to_chat_message(msg).model_dump_compat()
    chat_message_dict["attachments"] = attachments
    chat_message_dict["name"] = name
    chat_message_dict["id"] = msg.id
    # _convert_to_message from langchain_core.messages.utils expects an empty string instead of None
    if not chat_message_dict.get("content"):
        chat_message_dict["content"] = ""

    chat_agent_msg = ChatAgentMessage(**chat_message_dict)
    return chat_agent_msg.model_dump_compat(exclude_none=True)


class ChatAgentToolNode(ToolNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
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
        if isinstance(input, list):
            output_type = "list"
            message = input[-1]
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            output_type = "dict"
            message = messages[-1]
        elif messages := getattr(input, self.messages_key, None):
            # Assume dataclass-like state that can coerce from dict
            output_type = "dict"
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        tool_calls = [self._inject_tool_args(call, input, store) for call in message["tool_calls"]]
        return tool_calls, output_type


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent):
        self.agent = agent

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

    def predict_stream(
        self, model_input: list[ChatAgentMessage], params: Optional[ChatAgentParams] = None
    ):
        for event in self.agent.stream(
            {"messages": self._convert_messages_to_dict(model_input)}, stream_mode="updates"
        ):
            for node_data in event.values():
                if not node_data:
                    continue
                yield ChatAgentResponse(**node_data)
