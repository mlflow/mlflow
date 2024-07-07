from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from tests.llama_index._llama_index_test_fixtures import llama_index_patches


def test_llama_index_openai_chat_completions_patch():
    with llama_index_patches():
        from llama_index.llms.openai import OpenAI

        predicted = OpenAI().chat(messages=[ChatMessage(role="user", content="hi")])

        assert isinstance(predicted, ChatResponse)


def test_llama_index_openai_embedding_patch():
    with llama_index_patches():
        from llama_index.embeddings.openai import OpenAIEmbedding

        predicted = OpenAIEmbedding().get_query_embedding("hi")

        assert isinstance(predicted, list)
        assert all(isinstance(x, float) for x in predicted)
