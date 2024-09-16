from typing import Any, List, Literal, Optional

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import mlflow


def get_fake_chat_model(endpoint="fake-endpoint"):
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.chat_models import ChatDatabricks, ChatMlflow
    from langchain.schema.messages import BaseMessage
    from langchain_core.outputs import ChatResult

    class FakeChatModel(ChatDatabricks):
        """Fake Chat Model wrapper for testing purposes."""

        endpoint: str = "fake-endpoint"

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "test_content",
                        },
                        "finish_reason": None,
                    }
                ],
            }
            return ChatMlflow._create_chat_result(response)

        @property
        def _llm_type(self) -> str:
            return "fake chat model"

    return FakeChatModel(endpoint=endpoint)


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"


llm = get_fake_chat_model()
tools = [get_weather]
agent = create_react_agent(llm, tools)


def wrap_lg(input):
    if not isinstance(input, dict):
        if isinstance(input, list) and len(input) > 0:
            # Extract the content from the HumanMessage
            content = input[0].content.strip('"')
            input = {"messages": [{"role": "user", "content": content}]}
    return agent.invoke(input)


chain = RunnableLambda(wrap_lg)

mlflow.models.set_model(chain)
