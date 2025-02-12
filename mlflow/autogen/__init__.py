from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Define a tool that searches the web for information.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


# Create an agent that uses the OpenAI GPT-4o model.
# model_client = OpenAIChatCompletionClient(
#     model="gpt-4o",
#     # api_key="YOUR_API_KEY",
# )
# agent = AssistantAgent(
#     name="assistant",
#     model_client=model_client,
#     tools=[web_search],
#     system_message="Use tools to solve tasks.",
# )

from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.annotations import experimental

FLAVOR_NAME = "autogen"

@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    async def patch(original, self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        print("hello", result)
        return result

    safe_patch(FLAVOR_NAME, AssistantAgent, "on_messages_stream", patch)
    