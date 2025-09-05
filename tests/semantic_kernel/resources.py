import openai
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from tests.tracing.helper import reset_autolog_state  # noqa: F401


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
    from semantic_kernel.prompt_template import PromptTemplateConfig

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

    prompt_template_config = PromptTemplateConfig(
        template="{{$chat_history}}{{$user_input}}", allow_dangerously_set_content=True
    )

    chat_function = kernel.add_function(
        plugin_name="ChatBot",
        function_name="Chat",
        prompt_template_config=prompt_template_config,
        template_format="semantic-kernel",
        prompt_execution_settings=settings,
    )

    chat_history = ChatHistory(
        system_message=(
            "You are a chat bot named Mosscap, dedicated to figuring out what people need."
        )
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
        allow_dangerously_set_content=True,
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


async def _create_and_invoke_text_completion(mock_openai):
    """Test text completion methods - parser extracts {"prompt": "..."}"""
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    kernel = Kernel()
    kernel.add_service(
        OpenAITextCompletion(
            service_id="text-davinci",
            ai_model_id="text-davinci-003",
            async_client=openai_client,
        )
    )
    text_service = kernel.get_service("text-davinci")
    settings = kernel.get_prompt_execution_settings_from_service_id("text-davinci")
    return await text_service.get_text_content("Complete this: The sky is", settings)


async def _create_and_invoke_embeddings(mock_openai):
    """Test embedding methods - parser extracts {"texts": [...]}"""
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    embedding_service = OpenAITextEmbedding(
        service_id="embedding",
        ai_model_id="text-embedding-ada-002",
        async_client=openai_client,
    )
    texts = ["Hello world", "Semantic kernel", "MLflow tracing"]
    return await embedding_service.generate_embeddings(texts)


async def _create_and_invoke_chat_completion_direct(mock_openai):
    """Test direct chat completion - parser extracts {"messages": [...]}"""
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    chat_history = ChatHistory()
    chat_history.add_user_message("What is semantic kernel?")
    chat_history.add_assistant_message("Semantic Kernel is an AI orchestration framework.")
    chat_history.add_user_message("Tell me more about it.")

    chat_service = kernel.get_service("chat")
    settings = kernel.get_prompt_execution_settings_from_service_id("chat")
    return await chat_service.get_chat_message_content(chat_history, settings)


async def _create_and_invoke_kernel_function_object(mock_openai):
    """
    Test kernel.invoke with function object and arguments
    """
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    function = kernel.add_function(
        plugin_name="MathPlugin",
        function_name="Add",
        prompt="Add {{$num1}} and {{$num2}}",
        template_format="semantic-kernel",
    )

    return await kernel.invoke(function, KernelArguments(num1=5, num2=3))
