import openai
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from tests.tracing.helper import (
    reset_autolog_state,  # noqa: F401
)


async def _create_and_invoke_kernel_simple(mock_openai):
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )
    return await kernel.invoke_prompt("Is sushi the best food ever?")


async def _create_and_invoke_kernel_complex(mock_openai):
    # Set up kernel + OpenAI service
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    settings = kernel.get_prompt_execution_settings_from_service_id("chat-gpt")
    settings.max_tokens = 100
    settings.temperature = 0.7
    settings.top_p = 0.8

    chat_function = kernel.add_function(
        plugin_name="ChatBot",
        function_name="Chat",
        prompt="{{$chat_history}}{{$user_input}}",
        template_format="semantic-kernel",
        prompt_execution_settings=settings,
    )

    # Prepare input
    chat_history = ChatHistory(
        system_message="You are a chat bot named Mosscap, dedicated to figuring out what people need."
    )
    chat_history.add_user_message("Hi there, who are you?")
    chat_history.add_assistant_message(
        "I am Mosscap, a chat bot. I'm trying to figure out what people need."
    )
    user_input = "I want to find a hotel in Seattle with free wifi and a pool."

    return await kernel.invoke(
        chat_function,
        KernelArguments(
            user_input=user_input,
            chat_history=chat_history,
        ),
    )


async def _create_and_invoke_chat_agent(mock_openai):
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    service = OpenAIChatCompletion(
        service_id="chat-gpt",
        ai_model_id="gpt-4o-mini",
        async_client=openai_client,
    )
    agent = ChatCompletionAgent(
        service=service,
        name="sushi_agent",
        instructions="You are a master at all things sushi. But, you are not very smart.",
    )
    return await agent.get_response(messages="How do I make sushi?")
