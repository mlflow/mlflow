"""
This is an example for leveraging MLflow's auto tracing capabilities for Haystack.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
"""

from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.core.pipeline.async_pipeline import AsyncPipeline

import mlflow

# Turn on auto tracing for Haystack by calling mlflow.haystack.autolog()
# This will automatically trace:
# - Pipeline executions (both sync and async)
# - Individual component executions
# - Token usage for LLM components
# - Component metadata and parameters
mlflow.haystack.autolog()


# Example 1: Synchronous Pipeline
def sync_pipeline_example():
    print("=== Synchronous Pipeline Example ===")

    pipeline = Pipeline()
    llm = OpenAIGenerator(model="gpt-4o-mini")

    prompt_template = """Answer the question. {{question}}"""
    prompt_builder = PromptBuilder(template=prompt_template)

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder", "llm")

    question = "Who lives in Paris?"
    results = pipeline.run({"prompt_builder": {"question": question}})

    print("Question:", question)
    print("Answer:", results["llm"]["replies"][0])
    print()


# Example 2: Asynchronous Pipeline
async def async_pipeline_example():
    print("=== Asynchronous Pipeline Example ===")

    pipeline = AsyncPipeline()

    llm = OpenAIGenerator(model="gpt-4o-mini")
    prompt_template = """Tell me about: {{topic}}"""
    prompt_builder = PromptBuilder(template=prompt_template)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder", "llm")

    topic = "artificial intelligence"
    results = await pipeline.run_async({"prompt_builder": {"topic": topic}})

    print("Topic:", topic)
    print("Response:", results["llm"]["replies"][0])
    print()
