"""
This example demonstrates how to create a trace to track the execution of a multi-threaded application.

To trace a multi-threaded operation, you need to use the low-level MLflow client APIs to create a trace and spans, because the high-level fluent APIs are not thread-safe.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

import mlflow

exp = mlflow.set_experiment("mlflow-tracing-example")
exp_id = exp.experiment_id

client = openai.OpenAI()

# Enable MLflow Tracing for OpenAI
mlflow.openai.autolog()


@mlflow.trace
def worker(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=100,
    )
    return response.choices[0].message.content


@mlflow.trace
def main(questions: list[str]) -> list[str]:
    results = []
    # Almost same as how you would use ThreadPoolExecutor, but two additional steps
    #  1. Copy the context in the main thread using copy_context()
    #  2. Use ctx.run() to run the worker in the copied context
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for question in questions:
            ctx = contextvars.copy_context()
            futures.append(executor.submit(ctx.run, worker, question))
        for future in as_completed(futures):
            results.append(future.result())
    return results


questions = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

main(questions)

print(
    "\033[92m"
    + "🤖Now run `mlflow server` and open MLflow UI to see the trace visualization!"
    + "\033[0m"
)
