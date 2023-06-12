from .abc import AbstractProvider
import openai


class OpenAIProvider(AbstractProvider):
    NAME = "openai"
    SUPPORTED_ROUTES = ("chat", "completions", "embeddings")

    async def chat(self, payload):
        return await openai.ChatCompletion.acreate(...)

    async def completions(self, payload):
        return await openai.Completion.acreate(...)

    async def embeddings(self, payload):
        return await openai.Embedding.acreate(...)
